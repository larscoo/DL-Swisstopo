from __future__ import annotations

import csv
import io
import json
import math
import shutil
import threading
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from flask import Flask, abort, jsonify, render_template, request, send_file

from artifact_utils import (
    list_available_checkpoints,
    resolve_default_checkpoint_path,
    resolve_inference_threshold,
)
from train import build_model, default_transforms, resolve_device

WEBAPP_DIR = Path(__file__).resolve().parent
BASE_DIR = WEBAPP_DIR.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UNLABELED_DIR = DATA_DIR / "unlabeled"
POSITIVE_DIR = DATA_DIR / "y"
NEGATIVE_DIR = DATA_DIR / "n"
CSV_PATH = DATA_DIR / "labels.csv"
PAGE_SIZE = 16

app = Flask(__name__, template_folder="templates", static_folder="static")
_csv_lock = threading.Lock()
_model_lock = threading.Lock()
_rows: list[dict[str, str]] = []
_fieldnames: list[str] = []
_region_indices: dict[str, list[int]] = {}
_pages_by_region: dict[str, list[list[int]]] = {}
_predict_model = None
_predict_device = None
_predict_threshold = 0.5
_predict_image_size = 224
_predict_model_name = ""
_predict_transform = None
_predict_checkpoint_path: Path | None = None


def _read_json_file(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _checkpoint_metrics_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "metrics.json"


def _checkpoint_analysis_summary_path(checkpoint_path: Path) -> Path:
    return ARTIFACTS_DIR / "analysis" / checkpoint_path.parent.name / "summary.json"


def _best_history_entry(history: list[dict[str, object]], key: str) -> dict[str, object] | None:
    ranked: list[dict[str, object]] = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)):
            ranked.append(row)
    if not ranked:
        return None
    return max(ranked, key=lambda row: float(row.get(key, 0.0)))


def _build_model_stats_payload(checkpoint_path: Path) -> dict[str, object]:
    metrics_path = _checkpoint_metrics_path(checkpoint_path)
    summary_path = _checkpoint_analysis_summary_path(checkpoint_path)
    metrics_payload = _read_json_file(metrics_path)
    summary_payload = _read_json_file(summary_path)

    stats: dict[str, object] = {
        "available": metrics_payload is not None or summary_payload is not None,
        "metrics_path": metrics_path.relative_to(BASE_DIR).as_posix() if metrics_path.exists() else "",
        "analysis_summary_path": summary_path.relative_to(BASE_DIR).as_posix() if summary_path.exists() else "",
    }

    if metrics_payload is not None:
        history_raw = metrics_payload.get("history", [])
        history = [row for row in history_raw if isinstance(row, dict)]
        best_epoch = _best_history_entry(history, "val_f1")
        final_epoch = history[-1] if history else None
        stats.update(
            {
                "dataset_source": metrics_payload.get("dataset_source", ""),
                "epochs": len(history),
                "history": history,
                "best_epoch": best_epoch,
                "final_epoch": final_epoch,
                "model_info": metrics_payload.get("model_info", {}),
                "split_info": metrics_payload.get("split_info", {}),
                "training_config": metrics_payload.get("training_config", {}),
            }
        )

    if summary_payload is not None:
        stats.update(
            {
                "evaluation_split": summary_payload.get("split", ""),
                "evaluation_threshold": summary_payload.get("threshold"),
                "test_metrics": summary_payload.get("metrics", {}),
                "false_positives": summary_payload.get("false_positives"),
                "false_negatives": summary_payload.get("false_negatives"),
                "skipped_rows": summary_payload.get("skipped_rows"),
            }
        )

    return stats


def _model_status_payload(
    checkpoint_path: Path,
    device_type: str,
    threshold: float,
    image_size: int,
    model_name: str,
) -> dict[str, object]:
    return {
        "ready": True,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_relpath": checkpoint_path.relative_to(BASE_DIR).as_posix(),
        "device": device_type,
        "threshold": threshold,
        "image_size": image_size,
        "model_name": model_name,
        "available_checkpoints": _list_available_checkpoints(),
        "model_stats": _build_model_stats_payload(checkpoint_path),
    }


def _resolve_image_disk_path(image_path: str) -> Path:
    path = Path(str(image_path or "").strip())
    return path if path.is_absolute() else (BASE_DIR / path)


def _make_image_url(image_path: str) -> str:
    return f"/files/{str(image_path or '').lstrip('/')}"


def _parse_xy_from_stem(stem: str) -> tuple[int, int]:
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected filename stem '<x>_<y>', got: {stem!r}")
    return int(parts[0]), int(parts[1])


def _spatial_group(x: int, y: int, bin_size: int = 1000) -> str:
    return f"{x // bin_size}_{y // bin_size}"


def _row_is_pending(row: dict[str, str]) -> bool:
    return not str(row.get("label", "")).strip()


def _is_unlabeled_path(image_path: str) -> bool:
    normalized = str(image_path or "").strip().replace("\\", "/")
    return normalized.startswith("data/unlabeled/")


def _infer_region_id(path: Path, x: int, y: int) -> str:
    try:
        relative_parent = path.parent.relative_to(UNLABELED_DIR)
    except ValueError:
        relative_parent = Path(".")

    if relative_parent != Path("."):
        return f"region_{relative_parent.as_posix().replace('/', '_')}"
    return _spatial_group(x, y)


def _ensure_fieldnames() -> None:
    global _fieldnames
    if _fieldnames:
        return
    _fieldnames = ["image_path", "label", "region_id", "x", "y", "block_id"]


def _rebuild_pending_views() -> None:
    global _region_indices, _pages_by_region

    by_region: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(_rows):
        region = _normalize_region(row.get("region_id", ""))
        row["region_id"] = region
        if _row_is_pending(row):
            by_region[region].append(idx)

    _region_indices = dict(sorted(by_region.items(), key=lambda kv: kv[0]))
    _pages_by_region = {
        region: _build_pages_for_region(indices, page_size=PAGE_SIZE)
        for region, indices in _region_indices.items()
    }


def _sync_unlabeled_rows() -> bool:
    changed = False
    UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    pending_by_path = {
        str(row.get("image_path", "")).strip(): idx
        for idx, row in enumerate(_rows)
        if _row_is_pending(row)
    }
    seen_paths: set[str] = set()

    for path in sorted(UNLABELED_DIR.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        rel_path = path.relative_to(BASE_DIR).as_posix()
        seen_paths.add(rel_path)
        x, y = _parse_xy_from_stem(path.stem)
        region_id = _infer_region_id(path, x, y)

        if rel_path in pending_by_path:
            row = _rows[pending_by_path[rel_path]]
            if row.get("x") != str(x) or row.get("y") != str(y) or row.get("region_id") != region_id:
                row["x"] = str(x)
                row["y"] = str(y)
                row["region_id"] = region_id
                changed = True
            continue

        _rows.append(
            {
                "image_path": rel_path,
                "label": "",
                "region_id": region_id,
                "x": str(x),
                "y": str(y),
                "block_id": "",
            }
        )
        changed = True

    retained_rows: list[dict[str, str]] = []
    for row in _rows:
        image_path = str(row.get("image_path", "")).strip()
        if _row_is_pending(row) and _is_unlabeled_path(image_path) and image_path not in seen_paths:
            changed = True
            continue
        retained_rows.append(row)

    if len(retained_rows) != len(_rows):
        _rows[:] = retained_rows

    return changed


def _refresh_pending_queue() -> None:
    changed = _sync_unlabeled_rows()
    _rebuild_pending_views()
    if changed or not CSV_PATH.exists():
        write_csv()


def _move_labeled_file(row: dict[str, str], label: str) -> None:
    source = _resolve_image_disk_path(row.get("image_path", ""))
    if not source.exists():
        raise FileNotFoundError(f"Image file not found: {source}")

    target_dir = POSITIVE_DIR if label == "1" else NEGATIVE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name

    if target.exists() and target.resolve() != source.resolve():
        raise FileExistsError(f"Target file already exists: {target}")

    if target.resolve() != source.resolve():
        shutil.move(source.as_posix(), target.as_posix())

    row["image_path"] = target.relative_to(BASE_DIR).as_posix()
    row["label"] = label


def _assert_model_is_finite(model) -> None:
    invalid = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            invalid.append(name)
            if len(invalid) >= 3:
                break
    if invalid:
        raise ValueError(f"Checkpoint contains non-finite weights (for example: {', '.join(invalid)})")


def _list_available_checkpoints() -> list[dict[str, str]]:
    return list_available_checkpoints(BASE_DIR)


def _resolve_checkpoint_path(checkpoint_value: str | None) -> Path:
    requested = str(checkpoint_value or "").strip()
    if not requested:
        return resolve_default_checkpoint_path(BASE_DIR)

    path = Path(requested)
    candidate = path if path.is_absolute() else (BASE_DIR / path)
    candidate = candidate.resolve()
    try:
        candidate.relative_to(BASE_DIR)
    except ValueError as exc:
        raise ValueError(f"Checkpoint path is outside the workspace: {requested}") from exc
    return candidate


def _load_predict_model(checkpoint_value: str | None = None) -> dict[str, object]:
    global _predict_model, _predict_device, _predict_threshold, _predict_image_size, _predict_model_name, _predict_transform, _predict_checkpoint_path

    with _model_lock:
        checkpoint_path = _resolve_checkpoint_path(checkpoint_value)
        if _predict_model is not None and _predict_checkpoint_path is not None and _predict_checkpoint_path.resolve() == checkpoint_path.resolve():
            return _model_status_payload(
                _predict_checkpoint_path,
                _predict_device.type,
                _predict_threshold,
                _predict_image_size,
                _predict_model_name,
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_args = checkpoint.get("args", {})
        model_info = checkpoint.get("model_info", {})
        model_name = str(model_info.get("model_name") or model_args.get("model") or "simple_cnn")
        image_size = int(model_args.get("image_size", 224))
        threshold = resolve_inference_threshold(
            BASE_DIR,
            checkpoint_path,
            float(checkpoint.get("best_threshold", 0.5)),
        )

        device = resolve_device("auto")
        model, _ = build_model(model_name, "off")
        model.load_state_dict(checkpoint["model_state_dict"])
        _assert_model_is_finite(model)
        model = model.to(device)
        model.eval()

        _, eval_transform = default_transforms(image_size)
        _predict_model = model
        _predict_device = device
        _predict_threshold = threshold
        _predict_image_size = image_size
        _predict_model_name = model_name
        _predict_transform = eval_transform
        _predict_checkpoint_path = checkpoint_path

        return _model_status_payload(
            checkpoint_path,
            device.type,
            threshold,
            image_size,
            model_name,
        )


def _predict_image(image_bytes: bytes) -> tuple[float, int]:
    if _predict_model is None or _predict_transform is None or _predict_device is None:
        _load_predict_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _predict_transform(image).unsqueeze(0).to(_predict_device)
    with torch.no_grad():
        logits = _predict_model(tensor)
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        score = float(torch.sigmoid(logits).detach().cpu().item())
    if not math.isfinite(score):
        raise ValueError("Model returned a non-finite score. The checkpoint is likely corrupted and should be retrained.")
    prediction = 1 if score >= _predict_threshold else 0
    return score, prediction


def _to_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _spatial_sort_key(row: dict[str, str]) -> tuple[float, float]:
    # y descending (north -> south), x ascending (west -> east)
    y = _to_float(row.get("y", "0"))
    x = _to_float(row.get("x", "0"))
    return (-y, x)


def _morton_code(x: int, y: int) -> int:
    z = 0
    for i in range(32):
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z


def _build_pages_from_spatial(indices: list[int], page_size: int = PAGE_SIZE) -> list[list[int]]:
    if not indices:
        return []

    entries: list[tuple[int, int, int]] = []
    for idx in indices:
        row = _rows[idx]
        x = int(round(_to_float(row.get("x", "0"))))
        y = int(round(_to_float(row.get("y", "0"))))
        entries.append((idx, x, y))

    min_x = min(x for _, x, _ in entries)
    min_y = min(y for _, _, y in entries)

    ranked: list[tuple[int, int]] = []
    for idx, x, y in entries:
        gx = max((x - min_x) // 25, 0)
        gy = max((y - min_y) // 25, 0)
        ranked.append((_morton_code(gx, gy), idx))

    ranked.sort(key=lambda t: t[0])
    ordered_indices = [idx for _, idx in ranked]

    pages: list[list[int]] = []
    for i in range(0, len(ordered_indices), page_size):
        chunk = ordered_indices[i : i + page_size]
        chunk.sort(key=lambda idx: _spatial_sort_key(_rows[idx]))
        pages.append(chunk)

    return pages


def _build_pages_for_region(indices: list[int], page_size: int = PAGE_SIZE) -> list[list[int]]:
    by_block: dict[str, list[int]] = defaultdict(list)
    without_block: list[int] = []

    for idx in indices:
        block_id = str(_rows[idx].get("block_id", "")).strip()
        if block_id:
            by_block[block_id].append(idx)
        else:
            without_block.append(idx)

    pages: list[list[int]] = []

    if by_block:
        for block_id in sorted(by_block.keys()):
            block_indices = by_block[block_id]
            block_indices.sort(key=lambda idx: _spatial_sort_key(_rows[idx]))
            pages.append(block_indices)

    if without_block:
        pages.extend(_build_pages_from_spatial(without_block, page_size=page_size))

    return pages


def _normalize_region(value: str) -> str:
    region = str(value or "").strip()
    return region if region else "region_unknown"


def _resolve_region(requested_region: str) -> str:
    regions = sorted(_region_indices.keys())
    if not regions:
        return ""

    requested = str(requested_region or "").strip()
    if requested and requested in _region_indices:
        return requested

    return regions[0]


def load_csv() -> None:
    global _rows, _fieldnames, _region_indices, _pages_by_region

    with _csv_lock:
        if CSV_PATH.exists():
            with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                _fieldnames = reader.fieldnames or []
                _rows = list(reader)
        else:
            _fieldnames = []
            _rows = []

        _ensure_fieldnames()
        _refresh_pending_queue()


def write_csv() -> None:
    tmp_path = CSV_PATH.with_suffix(".tmp")

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames)
        writer.writeheader()
        writer.writerows(_rows)

    tmp_path.replace(CSV_PATH)


@app.get("/")
@app.get("/predict")
def index():
    return render_template("index.html")


@app.get("/files/<path:filename>")
def workspace_image_file(filename: str):
    candidate = (BASE_DIR / filename).resolve()
    try:
        candidate.relative_to(BASE_DIR)
    except ValueError:
        abort(404)

    if not candidate.is_file():
        abort(404)

    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        abort(404)

    return send_file(candidate)


@app.get("/api/regions")
def get_regions():
    with _csv_lock:
        _refresh_pending_queue()
        regions = []
        for region_id in sorted(_region_indices.keys()):
            indices = _region_indices[region_id]
            total = len(indices)
            pages = len(_pages_by_region.get(region_id, []))
            regions.append(
                {
                    "region_id": region_id,
                    "total": total,
                    "labeled": 0,
                    "unlabeled": total,
                    "pages": pages,
                }
            )

    return jsonify({"regions": regions})


@app.get("/api/page")
def get_page():
    page = max(int(request.args.get("page", 1)), 1)
    requested_region = request.args.get("region_id", "")

    with _csv_lock:
        _refresh_pending_queue()
        region_id = _resolve_region(requested_region)
        if not region_id:
            return jsonify(
                {
                    "region_id": "",
                    "page": 1,
                    "page_size": 0,
                    "total_items": 0,
                    "total_pages": 1,
                    "items": [],
                }
            )

        pages = _pages_by_region.get(region_id, [])
        total_items = len(_region_indices.get(region_id, []))
        total_pages = max(len(pages), 1)
        page = min(page, total_pages)
        page_indices = pages[page - 1] if pages else []

        items = []
        for idx in page_indices:
            row = _rows[idx]
            image_path = row.get("image_path", "")
            items.append(
                {
                    "index": idx,
                    "image_path": image_path,
                    "image_url": _make_image_url(image_path),
                    "file_exists": _resolve_image_disk_path(image_path).exists(),
                    "label": row.get("label", ""),
                }
            )

    return jsonify(
        {
            "region_id": region_id,
            "page": page,
            "page_size": len(page_indices),
            "total_items": total_items,
            "total_pages": total_pages,
            "items": items,
        }
    )


@app.post("/api/update")
def update_labels():
    data = request.get_json(silent=True) or {}
    updates = data.get("updates", [])

    if not isinstance(updates, list):
        return jsonify({"error": "'updates' muss eine Liste sein."}), 400

    with _csv_lock:
        _refresh_pending_queue()
        changed = 0
        for update in updates:
            if not isinstance(update, dict):
                continue

            index = update.get("index")
            label = update.get("label", "")

            if not isinstance(index, int):
                continue
            if index < 0 or index >= len(_rows):
                continue
            if str(label).strip() not in {"0", "1"}:
                continue

            _move_labeled_file(_rows[index], str(label).strip())
            changed += 1

        if changed:
            write_csv()
            _refresh_pending_queue()

    return jsonify({"updated": changed})


@app.get("/api/stats")
def stats():
    requested_region = request.args.get("region_id", "")

    with _csv_lock:
        _refresh_pending_queue()
        region_id = _resolve_region(requested_region)
        if not region_id:
            return jsonify({"region_id": "", "total": 0, "labeled": 0, "unlabeled": 0})

        indices = _region_indices.get(region_id, [])
        total = len(indices)

    return jsonify(
        {
            "region_id": region_id,
            "total": total,
            "labeled": 0,
            "unlabeled": total,
        }
    )


@app.get("/api/model-status")
def model_status():
    checkpoint_value = request.args.get("checkpoint", "")
    try:
        return jsonify(_load_predict_model(checkpoint_value))
    except Exception as exc:
        return jsonify(
            {
                "ready": False,
                "error": str(exc),
                "checkpoint_path": "",
                "available_checkpoints": _list_available_checkpoints(),
            }
        ), 500


@app.post("/api/predict")
def predict():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Keine Bilder hochgeladen."}), 400

    checkpoint_value = request.form.get("checkpoint", "")

    try:
        model_status_payload = _load_predict_model(checkpoint_value)
    except Exception as exc:
        return jsonify({"error": f"Modell konnte nicht geladen werden: {exc}"}), 500

    results = []
    for file_storage in files:
        filename = str(file_storage.filename or "").strip() or "upload"
        try:
            score, prediction = _predict_image(file_storage.read())
        except Exception as exc:
            results.append(
                {
                    "filename": filename,
                    "error": str(exc),
                }
            )
            continue

        results.append(
            {
                "filename": filename,
                "score": score,
                "prediction": prediction,
                "label": "Fussgängerstreifen" if prediction == 1 else "Kein Fussgängerstreifen",
                "threshold": _predict_threshold,
            }
        )

    return jsonify(
        {
            "results": results,
            "model": model_status_payload,
        }
    )


if __name__ == "__main__":
    load_csv()
    app.run(debug=True)
else:
    load_csv()
