from __future__ import annotations

import csv
import threading
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "labels.csv"
IMAGES_DIR = BASE_DIR / "images"
PAGE_SIZE = 16

app = Flask(__name__)
_csv_lock = threading.Lock()
_rows: list[dict[str, str]] = []
_fieldnames: list[str] = []
_region_indices: dict[str, list[int]] = {}
_pages_by_region: dict[str, list[list[int]]] = {}


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
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _fieldnames = reader.fieldnames or [
                "image_path",
                "label",
                "region_id",
                "x",
                "y",
                "block_id",
            ]
            _rows = list(reader)

        by_region: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(_rows):
            region = _normalize_region(row.get("region_id", ""))
            row["region_id"] = region
            by_region[region].append(idx)

        _region_indices = dict(sorted(by_region.items(), key=lambda kv: kv[0]))
        _pages_by_region = {
            region: _build_pages_for_region(indices, page_size=PAGE_SIZE)
            for region, indices in _region_indices.items()
        }


def write_csv() -> None:
    tmp_path = CSV_PATH.with_suffix(".tmp")

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames)
        writer.writeheader()
        writer.writerows(_rows)

    tmp_path.replace(CSV_PATH)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/images/<path:filename>")
def image_file(filename: str):
    return send_from_directory(IMAGES_DIR, filename)


@app.get("/api/regions")
def get_regions():
    with _csv_lock:
        regions = []
        for region_id in sorted(_region_indices.keys()):
            indices = _region_indices[region_id]
            total = len(indices)
            labeled = sum(1 for idx in indices if str(_rows[idx].get("label", "")).strip())
            pages = len(_pages_by_region.get(region_id, []))
            regions.append(
                {
                    "region_id": region_id,
                    "total": total,
                    "labeled": labeled,
                    "unlabeled": total - labeled,
                    "pages": pages,
                }
            )

    return jsonify({"regions": regions})


@app.get("/api/page")
def get_page():
    page = max(int(request.args.get("page", 1)), 1)
    requested_region = request.args.get("region_id", "")

    with _csv_lock:
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
            items.append(
                {
                    "index": idx,
                    "image_path": row.get("image_path", ""),
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

            _rows[index]["label"] = str(label).strip()
            changed += 1

        if changed:
            write_csv()

    return jsonify({"updated": changed})


@app.get("/api/stats")
def stats():
    requested_region = request.args.get("region_id", "")

    with _csv_lock:
        region_id = _resolve_region(requested_region)
        if not region_id:
            return jsonify({"region_id": "", "total": 0, "labeled": 0, "unlabeled": 0})

        indices = _region_indices.get(region_id, [])
        total = len(indices)
        labeled = sum(1 for idx in indices if str(_rows[idx].get("label", "")).strip())

    return jsonify(
        {
            "region_id": region_id,
            "total": total,
            "labeled": labeled,
            "unlabeled": total - labeled,
        }
    )


if __name__ == "__main__":
    load_csv()
    app.run(debug=True)
else:
    load_csv()
