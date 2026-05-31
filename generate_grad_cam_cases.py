#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_errors import resolve_image_path
from artifact_utils import resolve_default_checkpoint_path
from grad_cam import (
    compute_grad_cam,
    conv_layer_names,
    default_input_image,
    load_checkpoint_model,
    progress,
    resolve_path,
    resolve_target_layer,
    resize_cam_to_image,
    save_grad_cam_figure,
)
from train import default_transforms
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for false positives, false negatives and uncertain cases."
    )
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional all_predictions.csv path. Defaults to artifacts/analysis/<run-name>/all_predictions.csv.",
    )
    parser.add_argument("--layer", default=None, help="Optional Conv2d layer name to use.")
    parser.add_argument("--count", type=int, default=10, help="Number of samples per category. Defaults to 10.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to plots/grad_cam/<run-name>/interesting_cases/.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print available Conv2d layer names and exit.",
    )
    return parser.parse_args()


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def safe_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def select_cases(rows: list[dict[str, str]], count: int) -> dict[str, list[dict[str, str]]]:
    false_positives = [row for row in rows if str(row.get("error_type", "")).strip() == "false_positive"]
    false_negatives = [row for row in rows if str(row.get("error_type", "")).strip() == "false_negative"]
    uncertain = [
        row
        for row in rows
        if "score" in row and "threshold" in row and row.get("label", "") in {"0", "1"}
    ]

    false_positives.sort(key=lambda row: safe_float(str(row.get("score", "0"))), reverse=True)
    false_negatives.sort(key=lambda row: safe_float(str(row.get("score", "0"))))
    uncertain.sort(key=lambda row: abs(safe_float(str(row.get("score", "0"))) - safe_float(str(row.get("threshold", "0.5")))))

    return {
        "false_positives": false_positives[:count],
        "false_negatives": false_negatives[:count],
        "uncertain": uncertain[:count],
    }


def output_filename(row: dict[str, str], layer_name: str) -> str:
    stem = Path(str(row.get("image_path") or row.get("resolved_image_path") or "sample")).stem
    score = safe_float(str(row.get("score", "0")))
    label = str(row.get("label", ""))
    prediction = str(row.get("prediction", ""))
    return f"{stem}_y{label}_p{prediction}_s{score:.3f}_{layer_name.replace('.', '_')}.png"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    progress(1, 6, "Resolving checkpoint and predictions CSV")
    checkpoint_path = resolve_path(project_root, args.checkpoint, resolve_default_checkpoint_path(project_root))
    run_name = checkpoint_path.parent.name
    predictions_path = resolve_path(
        project_root,
        args.predictions,
        project_root / "artifacts" / "analysis" / run_name / "all_predictions.csv",
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    progress(2, 6, "Loading model and selecting target layer")
    model, model_info, image_size = load_checkpoint_model(checkpoint_path)
    available_layers = conv_layer_names(model)
    if args.list_layers:
        print("\n".join(available_layers))
        return
    layer_name, target_module = resolve_target_layer(model, args.layer)
    _, eval_transform = default_transforms(image_size)

    progress(3, 6, "Loading prediction rows and selecting cases")
    rows = load_prediction_rows(predictions_path)
    grouped = select_cases(rows, max(1, args.count))

    progress(4, 6, "Preparing output directories")
    output_dir = resolve_path(
        project_root,
        args.output_dir,
        project_root / "plots" / "grad_cam" / run_name / "interesting_cases",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = str(model_info.get("model_name") or run_name)
    total_images = sum(len(items) for items in grouped.values())
    processed = 0

    progress(5, 6, f"Generating Grad-CAMs for {total_images} selected images")
    for category, items in grouped.items():
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for row in items:
            raw_path = str(row.get("resolved_image_path") or row.get("image_path") or "").strip()
            image_path = resolve_image_path(project_root, raw_path)
            original_image = Image.open(image_path).convert("RGB")
            image_tensor = eval_transform(original_image)
            cam, probability = compute_grad_cam(model, target_module, image_tensor)
            cam_resized = resize_cam_to_image(cam, original_image.size)
            output_path = category_dir / output_filename(row, layer_name)
            save_grad_cam_figure(output_path, original_image, cam_resized, layer_name, checkpoint_name, probability)
            processed += 1
            print(f"[grad_cam_cases] {processed}/{total_images} saved: {output_path.name}", flush=True)

    progress(6, 6, f"Done. Grad-CAM cases written to {output_dir}")


if __name__ == "__main__":
    main()
