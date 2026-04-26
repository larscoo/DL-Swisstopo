#!/usr/bin/env python3

# python3 download_tiles.py

from __future__ import annotations
import argparse
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
OUTPUT_LABELS_CSV = "data/labels.csv"
OUTPUT_DIR = "data/unlabeled"


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


def normalize_region(region_id: str) -> str:
    region = str(region_id or "").strip()
    return region if region else "region_unknown"


def detect_existing_image_paths(labels_path: Path) -> set[str]:
    existing_paths: set[str] = set()
    if not labels_path.exists():
        return existing_paths

    with labels_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            image_path = str(row.get("image_path", "")).strip()
            if image_path:
                existing_paths.add(image_path)

    return existing_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download image tiles and write labels CSV.")
    parser.add_argument("--input-csv", default=INPUT_CSV, help="Input points CSV.")
    parser.add_argument("--output-labels-csv", default=OUTPUT_LABELS_CSV, help="Output labels CSV.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for downloaded images.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite labels CSV instead of appending. Default is append-safe mode.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    labels_path = Path(args.output_labels_csv)
    append_mode = not args.overwrite

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file '{input_path}' not found. "
            "Create it with columns: x,y,region_id (label optional)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        required = {"x", "y", "region_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns in {input_path}: {sorted(missing)}. "
                "Expected: x,y,region_id (optional: label)"
            )

        rows = list(reader)
        total = len(rows)
        if total == 0:
            raise ValueError(f"No rows found in {input_path}. Add at least one point.")

    existing_image_paths = detect_existing_image_paths(labels_path) if append_mode else set()
    write_header = True
    csv_mode = "w"
    if append_mode and labels_path.exists():
        write_header = labels_path.stat().st_size == 0
        csv_mode = "a"

    with labels_path.open(csv_mode, newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        if write_header:
            writer.writerow(["image_path", "label", "region_id", "x", "y", "block_id"])

        with requests.Session() as session:
            start_time = time.time()
            for i, row in enumerate(rows, start=1):
                x = float(row["x"])
                y = float(row["y"])
                x_int = int(round(x))
                y_int = int(round(y))
                region_id = normalize_region(row.get("region_id", ""))
                region_dir = output_dir / region_id
                region_dir.mkdir(parents=True, exist_ok=True)

                image_name = f"{x_int}_{y_int}.jpg"
                image_path = region_dir / image_name
                image_path_str = image_path.as_posix()

                if append_mode and image_path_str in existing_image_paths:
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
                    continue

                download_tile(session, x, y, image_path)

                writer.writerow(
                    [
                        image_path_str,
                        row.get("label", ""),
                        region_id,
                        x,
                        y,
                        row.get("block_id", ""),
                    ]
                )
                existing_image_paths.add(image_path_str)
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

    mode_text = "appended to" if append_mode else "overwrote"
    print(f"Done. Downloaded tiles into '{output_dir}/<region_id>/' and {mode_text} '{labels_path}'.")


if __name__ == "__main__":
    main()
