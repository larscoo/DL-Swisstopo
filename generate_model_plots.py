#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from artifact_utils import resolve_default_checkpoint_path

MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "matplotlib is required for generate_model_plots.py. "
        "Install it with: .venv/bin/python -m pip install matplotlib"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate confusion matrix, PR, ROC and training curves as PNG.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to the preferred checkpoint.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional all_predictions.csv path. Defaults to artifacts/analysis/<run-name>/all_predictions.csv.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional metrics.json path. Defaults to the checkpoint sibling metrics.json.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest CSV. Used for auto-generating predictions when analysis is missing.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Manifest split to analyze when predictions need to be generated. Defaults to 'test'.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device for auto-generating predictions. Defaults to 'auto'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for auto-generating predictions.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for auto-generating predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override when predictions need to be generated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for rows analyzed when predictions need to be generated.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to plots/model_plots/<run-name>/.",
    )
    return parser.parse_args()


def progress(step: int, total: int, message: str) -> None:
    print(f"[{step}/{total}] {message}", flush=True)


def resolve_path(project_root: Path, raw_value: str | None, fallback: Path) -> Path:
    if not raw_value:
        return fallback.resolve()
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def load_predictions(path: Path) -> tuple[list[int], list[float], list[int], float]:
    labels: list[int] = []
    scores: list[float] = []
    predictions: list[int] = []
    threshold = 0.5

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_raw = str(row.get("label", "")).strip()
            score_raw = str(row.get("score", "")).strip()
            pred_raw = str(row.get("prediction", "")).strip()
            threshold_raw = str(row.get("threshold", "")).strip()
            if label_raw not in {"0", "1"} or pred_raw not in {"0", "1"}:
                continue
            try:
                score = float(score_raw)
            except ValueError:
                continue
            labels.append(int(label_raw))
            scores.append(score)
            predictions.append(int(pred_raw))
            if threshold_raw:
                try:
                    threshold = float(threshold_raw)
                except ValueError:
                    pass

    if not labels:
        raise ValueError(f"No usable predictions found in {path}")
    return labels, scores, predictions, threshold


def load_history(path: Path) -> list[dict[str, float | int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if not isinstance(history, list):
        return []
    return [row for row in history if isinstance(row, dict)]


def ensure_predictions_exist(
    project_root: Path,
    checkpoint_path: Path,
    predictions_path: Path,
    *,
    manifest_path: Path | None,
    split: str,
    device: str,
    batch_size: int,
    num_workers: int,
    threshold: float | None,
    limit: int | None,
) -> None:
    if predictions_path.exists():
        return

    analysis_dir = predictions_path.parent
    analysis_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(project_root / "analyze_errors.py"),
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--output-dir",
        analysis_dir.as_posix(),
        "--split",
        split,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]
    if manifest_path is not None:
        command.extend(["--manifest", manifest_path.as_posix()])
    if threshold is not None:
        command.extend(["--threshold", str(threshold)])
    if limit is not None:
        command.extend(["--limit", str(limit)])

    print(
        f"Predictions missing for {checkpoint_path.parent.name}. "
        f"Generating analysis in {analysis_dir} from split={split!r} ...",
        flush=True,
    )
    subprocess.run(command, check=True, cwd=project_root)

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file could not be generated automatically: {predictions_path}"
        )


def confusion_counts(labels: list[int], predictions: list[int]) -> dict[str, int]:
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_precision_recall_curve(labels: list[int], scores: list[float]) -> tuple[list[tuple[float, float]], float]:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    total_pos = sum(labels)
    if total_pos == 0:
        return [(0.0, 1.0)], 0.0

    tp = 0
    fp = 0
    points: list[tuple[float, float]] = [(0.0, 1.0)]
    idx = 0
    while idx < len(pairs):
        score = pairs[idx][0]
        while idx < len(pairs) and pairs[idx][0] == score:
            if pairs[idx][1] == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        recall = tp / total_pos
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        points.append((recall, precision))

    if points[-1][0] < 1.0:
        points.append((1.0, points[-1][1]))

    area = 0.0
    for (recall_a, precision_a), (recall_b, precision_b) in zip(points, points[1:]):
        area += (recall_b - recall_a) * ((precision_a + precision_b) / 2.0)
    return points, area


def compute_roc_curve(labels: list[int], scores: list[float]) -> tuple[list[tuple[float, float]], float]:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return [(0.0, 0.0), (1.0, 1.0)], 0.5

    tp = 0
    fp = 0
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    idx = 0
    while idx < len(pairs):
        score = pairs[idx][0]
        while idx < len(pairs) and pairs[idx][0] == score:
            if pairs[idx][1] == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        points.append((fpr, tpr))

    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))

    area = 0.0
    for (fpr_a, tpr_a), (fpr_b, tpr_b) in zip(points, points[1:]):
        area += (fpr_b - fpr_a) * ((tpr_a + tpr_b) / 2.0)
    return points, area


def operating_point(counts: dict[str, int]) -> tuple[float, float, float, float]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return precision, recall, tpr, fpr


def history_series(history: list[dict[str, float | int]], key: str) -> tuple[list[float], list[float]]:
    epochs: list[float] = []
    values: list[float] = []
    for row in history:
        epoch = row.get("epoch")
        value = row.get(key)
        if isinstance(epoch, (int, float)) and isinstance(value, (int, float)):
            epochs.append(float(epoch))
            values.append(float(value))
    return epochs, values


def configure_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_confusion_matrix_png(path: Path, counts: dict[str, int], threshold: float, total: int) -> None:
    matrix = [
        [counts["tn"], counts["fp"]],
        [counts["fn"], counts["tp"]],
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(f"Confusion Matrix\nthreshold={threshold:.2f}, n={total}")

    max_value = max(counts.values()) if counts else 1
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            text_color = "white" if value > max_value / 2 else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=text_color, fontsize=12)

    save_figure(fig, path)


def generate_curve_png(
    path: Path,
    *,
    title: str,
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    auc_label: str,
    auc_value: float,
    operating_marker: tuple[float, float] | None = None,
    baseline: tuple[list[float], list[float]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, linewidth=2)
    if baseline is not None:
        ax.plot(baseline[0], baseline[1], linestyle="--", linewidth=1.5, color="gray")
    if operating_marker is not None:
        ax.scatter([operating_marker[0]], [operating_marker[1]], color="red", s=30, zorder=3)
        ax.annotate("threshold", operating_marker, xytext=(6, 6), textcoords="offset points", fontsize=9)

    configure_axes(ax, f"{title}\n{auc_label}={auc_value:.4f}", x_label, y_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    save_figure(fig, path)


def generate_training_curves_png(path: Path, history: list[dict[str, float | int]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_specs = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_accuracy", "val_accuracy"),
        ("F1", "train_f1", "val_f1"),
        ("PR AUC", "train_pr_auc", "val_pr_auc"),
    ]

    for ax, (title, train_key, val_key) in zip(axes.flat, plot_specs):
        train_epochs, train_values = history_series(history, train_key)
        val_epochs, val_values = history_series(history, val_key)
        if train_epochs:
            ax.plot(train_epochs, train_values, label="train", linewidth=2)
        if val_epochs:
            ax.plot(val_epochs, val_values, label="val", linewidth=2)
        configure_axes(ax, title, "epoch", "value")
        ax.legend()

    save_figure(fig, path)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    progress(1, 6, "Resolving inputs")
    checkpoint_path = resolve_path(
        project_root,
        args.checkpoint,
        resolve_default_checkpoint_path(project_root),
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_name = checkpoint_path.parent.name
    predictions_path = resolve_path(
        project_root,
        args.predictions,
        project_root / "artifacts" / "analysis" / run_name / "all_predictions.csv",
    )
    metrics_path = resolve_path(
        project_root,
        args.metrics,
        checkpoint_path.parent / "metrics.json",
    )
    manifest_path = resolve_path(
        project_root,
        args.manifest,
        checkpoint_path.parent / "manifest.csv",
    )
    output_dir = resolve_path(
        project_root,
        args.output_dir,
        project_root / "plots" / "model_plots" / run_name,
    )

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not manifest_path.exists() and not predictions_path.exists():
        raise FileNotFoundError(
            f"Neither predictions nor manifest found. Missing manifest: {manifest_path}"
        )

    progress(2, 6, "Ensuring prediction CSV exists")
    ensure_predictions_exist(
        project_root,
        checkpoint_path,
        predictions_path,
        manifest_path=manifest_path if manifest_path.exists() else None,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        limit=args.limit,
    )

    progress(3, 6, "Loading predictions and training history")
    labels, scores, predictions, threshold = load_predictions(predictions_path)
    history = load_history(metrics_path)

    progress(4, 6, "Computing metrics and curves")
    counts = confusion_counts(labels, predictions)
    pr_points, pr_auc = compute_precision_recall_curve(labels, scores)
    roc_points, roc_auc = compute_roc_curve(labels, scores)
    precision, recall, tpr, fpr = operating_point(counts)
    positive_rate = sum(labels) / len(labels)

    progress(5, 6, "Rendering PNG files with matplotlib")
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_confusion_matrix_png(output_dir / "confusion_matrix.png", counts, threshold, len(labels))
    generate_curve_png(
        output_dir / "pr_curve.png",
        title="Precision Recall Curve",
        x_values=[point[0] for point in pr_points],
        y_values=[point[1] for point in pr_points],
        x_label="Recall",
        y_label="Precision",
        auc_label="PR AUC",
        auc_value=pr_auc,
        operating_marker=(recall, precision),
        baseline=([0.0, 1.0], [positive_rate, positive_rate]),
    )
    generate_curve_png(
        output_dir / "roc_curve.png",
        title="ROC Curve",
        x_values=[point[0] for point in roc_points],
        y_values=[point[1] for point in roc_points],
        x_label="False Positive Rate",
        y_label="True Positive Rate",
        auc_label="ROC AUC",
        auc_value=roc_auc,
        operating_marker=(fpr, tpr),
        baseline=([0.0, 1.0], [0.0, 1.0]),
    )
    generate_training_curves_png(output_dir / "training_curves.png", history)

    progress(6, 6, f"Done. PNG files written to {output_dir}")


if __name__ == "__main__":
    main()
