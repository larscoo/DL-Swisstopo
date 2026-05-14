#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from artifact_utils import resolve_default_checkpoint_path, resolve_inference_threshold
from train import build_model, default_transforms, resolve_device


class ManifestDataset(Dataset[tuple[torch.Tensor, str, int]]):
    def __init__(self, rows: list[dict[str, str]], transform, project_root: Path) -> None:
        self.rows = rows
        self.transform = transform
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        row = self.rows[index]
        image_path = resolve_image_path(self.project_root, row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path.as_posix(), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect false positives and false negatives for a trained run.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a model checkpoint. Defaults to the preferred checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to a manifest CSV. Defaults to the checkpoint sibling manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to artifacts/analysis/<run-name>/.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--split",
        default="test",
        help="Manifest split to analyze. Defaults to 'test'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the decision threshold. Default uses the checkpoint threshold.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of manifest rows to score after filtering by split.",
    )
    return parser.parse_args()


def resolve_image_path(project_root: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip())
    if candidate.is_absolute() and candidate.exists():
        return candidate

    repo_name = project_root.name
    parts = candidate.parts

    if candidate.is_absolute() and repo_name in parts:
        repo_index = parts.index(repo_name)
        remapped = project_root / Path(*parts[repo_index + 1 :])
        if remapped.exists():
            return remapped.resolve()

    if candidate.is_absolute() and "data" in parts:
        data_index = parts.index("data")
        remapped = project_root / Path(*parts[data_index:])
        if remapped.exists():
            return remapped.resolve()

    if not candidate.is_absolute():
        remapped = (project_root / candidate).resolve()
        if remapped.exists():
            return remapped

    raise FileNotFoundError(f"Could not resolve image path: {raw_path}")


def load_manifest_rows(
    manifest_path: Path,
    split_name: str,
    limit: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_value = str(row.get("split", "")).strip()
            label_value = str(row.get("label", "")).strip()
            if split_value != split_name:
                continue
            if label_value not in {"0", "1"}:
                skipped_rows.append(
                    {
                        "reason": "missing_or_invalid_label",
                        "image_path": str(row.get("image_path", "")),
                        "split": split_value,
                        "label": label_value,
                    }
                )
                continue
            selected_rows.append(dict(row))
            if limit is not None and len(selected_rows) >= limit:
                break

    return selected_rows, skipped_rows


def make_loader(
    rows: list[dict[str, str]],
    transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    project_root: Path,
) -> DataLoader:
    dataset = ManifestDataset(rows, transform, project_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def safe_float(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError(f"Encountered non-finite score: {value}")
    return float(value)


def calculate_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    tp = sum(1 for row in rows if row["prediction"] == 1 and row["label"] == 1)
    tn = sum(1 for row in rows if row["prediction"] == 0 and row["label"] == 0)
    fp = sum(1 for row in rows if row["prediction"] == 1 and row["label"] == 0)
    fn = sum(1 for row in rows if row["prediction"] == 0 and row["label"] == 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0

    return {
        "count": len(rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    checkpoint_path = resolve_default_checkpoint_path(project_root) if not args.checkpoint else (
        Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else (project_root / args.checkpoint)
    )
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    manifest_path = (
        checkpoint_path.parent / "manifest.csv"
        if not args.manifest
        else (Path(args.manifest) if Path(args.manifest).is_absolute() else (project_root / args.manifest))
    )
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint.get("args", {})
    model_info = checkpoint.get("model_info", {})
    model_name = str(model_info.get("model_name") or model_args.get("model") or "simple_cnn")
    image_size = int(model_args.get("image_size", 224))
    threshold = float(
        args.threshold
        if args.threshold is not None
        else resolve_inference_threshold(
            project_root,
            checkpoint_path.resolve(),
            float(checkpoint.get("best_threshold", 0.5)),
        )
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (project_root / "artifacts" / "analysis" / checkpoint_path.parent.name)
    )
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, _ = build_model(model_name, "off")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _, eval_transform = default_transforms(image_size)
    manifest_rows, skipped_rows = load_manifest_rows(manifest_path, args.split, args.limit)
    if not manifest_rows:
        raise ValueError(f"No manifest rows found for split={args.split!r} in {manifest_path}")

    existing_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        try:
            resolve_image_path(project_root, row["image_path"])
        except FileNotFoundError:
            skipped_rows.append(
                {
                    "reason": "missing_image",
                    "image_path": str(row.get("image_path", "")),
                    "split": str(row.get("split", "")),
                    "label": str(row.get("label", "")),
                }
            )
            continue
        existing_rows.append(row)

    if not existing_rows:
        raise ValueError("No existing image files remained after manifest filtering.")

    loader = make_loader(existing_rows, eval_transform, args.batch_size, args.num_workers, device, project_root)

    prediction_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for images, batch_paths, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()

            for image_path, row_index, score in zip(batch_paths, batch_indices.tolist(), probs):
                row = dict(existing_rows[row_index])
                score_value = safe_float(score)
                label_value = int(str(row.get("label", "0")))
                prediction_value = 1 if score_value >= threshold else 0
                row.update(
                    {
                        "resolved_image_path": image_path,
                        "score": f"{score_value:.6f}",
                        "threshold": f"{threshold:.4f}",
                        "prediction": prediction_value,
                        "label": label_value,
                        "error_type": (
                            "false_positive"
                            if prediction_value == 1 and label_value == 0
                            else "false_negative"
                            if prediction_value == 0 and label_value == 1
                            else "correct"
                        ),
                    }
                )
                prediction_rows.append(row)

    prediction_rows.sort(
        key=lambda row: (str(row["error_type"]), -float(str(row["score"])), str(row["resolved_image_path"]))
    )
    false_positives = [row for row in prediction_rows if row["error_type"] == "false_positive"]
    false_negatives = [row for row in prediction_rows if row["error_type"] == "false_negative"]

    write_rows(output_dir / "all_predictions.csv", prediction_rows)
    write_rows(output_dir / "false_positives.csv", false_positives)
    write_rows(output_dir / "false_negatives.csv", false_negatives)
    if skipped_rows:
        write_rows(output_dir / "skipped_rows.csv", skipped_rows)

    try:
        manifest_relpath = manifest_path.relative_to(project_root).as_posix()
    except ValueError:
        manifest_relpath = manifest_path.as_posix()

    summary = {
        "checkpoint": checkpoint_path.relative_to(project_root).as_posix(),
        "manifest": manifest_relpath,
        "split": args.split,
        "threshold": threshold,
        "image_size": image_size,
        "model_name": model_name,
        "device": device.type,
        "metrics": calculate_metrics(prediction_rows),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "skipped_rows": len(skipped_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Analyzed {len(prediction_rows)} rows from split={args.split!r} using {checkpoint_path.parent.name}.")
    print(
        f"Precision={summary['metrics']['precision']:.4f}, "
        f"Recall={summary['metrics']['recall']:.4f}, "
        f"F1={summary['metrics']['f1']:.4f}, "
        f"FP={len(false_positives)}, FN={len(false_negatives)}."
    )
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
