#!/usr/bin/env python3

# python3 download_tiles.py

from __future__ import annotations
import csv
import time
from pathlib import Path
import requests

WMS_URL = "https://wms.geo.admin.ch/"
WMS_LAYER = "ch.swisstopo.swissimage"
CRS = "EPSG:2056"
TILE_SIZE_M = 25.0
IMG_SIZE_PX = 256
REQUEST_DELAY_S = 0.1
INPUT_CSV = "points.csv"
OUTPUT_LABELS_CSV = "labels.csv"
OUTPUT_DIR = "images"

def build_bbox(x: float, y: float, tile_size_m: float) -> str:
    half = tile_size_m / 2.0
    return f"{x - half},{y - half},{x + half},{y + half}"

def download_tile(session: requests.Session, x: float, y: float, out_path: Path) -> None:
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": WMS_LAYER,
        "STYLES": "default",
        "CRS": CRS,
        "BBOX": build_bbox(x, y, TILE_SIZE_M),
        "WIDTH": str(IMG_SIZE_PX),
        "HEIGHT": str(IMG_SIZE_PX),
        "FORMAT": "image/jpeg",
    }
    resp = session.get(WMS_URL, params=params, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)

def main() -> None:
    input_path = Path(INPUT_CSV)
    output_dir = Path(OUTPUT_DIR)
    labels_path = Path(OUTPUT_LABELS_CSV)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file '{INPUT_CSV}' not found. "
            "Create it with columns: x,y,region_id (label optional)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        required = {"x", "y", "region_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns in {INPUT_CSV}: {sorted(missing)}. "
                "Expected: x,y,region_id (optional: label)"
            )

        rows = list(reader)
        total = len(rows)
        if total == 0:
            raise ValueError(f"No rows found in {INPUT_CSV}. Add at least one point.")

    with labels_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["image_path", "label", "region_id", "x", "y"])

        with requests.Session() as session:
            start_time = time.time()
            for i, row in enumerate(rows, start=1):
                x = float(row["x"])
                y = float(row["y"])
                image_name = f"tile_{i:05d}.jpg"
                image_path = output_dir / image_name

                download_tile(session, x, y, image_path)

                writer.writerow(
                    [
                        str(image_path.as_posix()),
                        row.get("label", ""),
                        row["region_id"],
                        x,
                        y,
                    ]
                )
                time.sleep(REQUEST_DELAY_S)

                elapsed = time.time() - start_time
                avg = elapsed / i
                eta = avg * (total - i)
                pct = (i / total) * 100.0
                print(
                    f"\rProgress: {i}/{total} ({pct:5.1f}%) | "
                    f"Elapsed: {elapsed:6.1f}s | ETA: {eta:6.1f}s",
                    end="",
                    flush=True,
                )

    print()

    print(f"Done. Downloaded tiles into '{OUTPUT_DIR}/' and wrote '{OUTPUT_LABELS_CSV}'.")


if __name__ == "__main__":
    main()
