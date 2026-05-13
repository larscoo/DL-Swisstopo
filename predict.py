#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from artifact_utils import resolve_default_checkpoint_path
from train import IMAGE_EXTENSIONS, build_model, default_transforms, resolve_device


class PredictDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained crosswalk classifier.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a model checkpoint. Defaults to the preferred checkpoint found under artifacts/.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory containing tiles to score.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/predictions/predictions.csv",
        help="CSV path for prediction results.",
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
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the decision threshold. Default uses the checkpoint threshold.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directories recursively.",
    )
    return parser.parse_args()


def collect_image_paths(input_path: Path, recursive: bool) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file extension: {input_path.suffix}")
        return [input_path.resolve()]

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    image_paths = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported image files found in: {input_path}")
    return image_paths


def make_loader(
    image_paths: list[Path],
    transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = PredictDataset(image_paths, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    checkpoint_path = resolve_default_checkpoint_path(project_root) if not args.checkpoint else Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint.get("args", {})
    model_info = checkpoint.get("model_info", {})
    model_name = str(model_info.get("model_name") or model_args.get("model") or "simple_cnn")
    pretrained = "off"
    image_size = int(model_args.get("image_size", 224))
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("best_threshold", 0.5))

    device = resolve_device(args.device)
    model, _ = build_model(model_name, pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    invalid = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            invalid.append(name)
            if len(invalid) >= 3:
                break
    if invalid:
        raise ValueError(
            "Checkpoint contains non-finite weights "
            f"(for example: {', '.join(invalid)}). Retrain the model before running prediction."
        )
    model = model.to(device)
    model.eval()

    _, eval_transform = default_transforms(image_size)
    image_paths = collect_image_paths(Path(args.input), recursive=args.recursive)
    loader = make_loader(image_paths, eval_transform, args.batch_size, args.num_workers, device)

    rows: list[tuple[str, float, int]] = []
    with torch.no_grad():
        for images, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            for image_path, score in zip(batch_paths, probs):
                if not math.isfinite(float(score)):
                    raise ValueError(
                        "Model returned a non-finite score. The checkpoint is likely corrupted and should be retrained."
                    )
                prediction = 1 if score >= threshold else 0
                rows.append((image_path, float(score), prediction))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "score", "prediction", "threshold"])
        for image_path, score, prediction in rows:
            writer.writerow([image_path, f"{score:.6f}", prediction, f"{threshold:.4f}"])

    positives = sum(prediction for _, _, prediction in rows)
    print(
        f"Scored {len(rows)} images on {device.type} with threshold={threshold:.4f}. "
        f"Predicted positives: {positives}."
    )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
