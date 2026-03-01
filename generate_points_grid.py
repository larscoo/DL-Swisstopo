#!/usr/bin/env python3

# python3 generate_points_grid.py --preset glarus --max-points 500 --region-id region_glarus_01 --output points.csv

from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path

PRESETS = {
    # Approximate bounding box around Glarus town (EPSG:2056).
    "glarus": (2722500.0, 1207000.0, 2725500.0, 1209500.0),
}

def frange(start: float, stop: float, step: float):
    value = start
    while value <= stop:
        yield value
        value += step

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grid points CSV.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()))
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--step", type=float, default=25.0, help="Grid spacing in meters.")
    parser.add_argument("--max-points", type=int, default=300, help="Random sample cap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--region-id", default="region_glarus_01")
    parser.add_argument("--output", default="points.csv")
    args = parser.parse_args()

    if args.preset:
        xmin, ymin, xmax, ymax = PRESETS[args.preset]
    else:
        values = (args.xmin, args.ymin, args.xmax, args.ymax)
        if any(v is None for v in values):
            raise SystemExit(
                "Provide either --preset or all of --xmin --ymin --xmax --ymax."
            )
        xmin, ymin, xmax, ymax = values

    if args.step <= 0:
        raise SystemExit("--step must be > 0")
    if xmin >= xmax or ymin >= ymax:
        raise SystemExit("Invalid bbox: expected xmin<xmax and ymin<ymax")

    points = [(x, y) for x in frange(xmin, xmax, args.step) for y in frange(ymin, ymax, args.step)]

    if args.max_points and len(points) > args.max_points:
        random.seed(args.seed)
        points = random.sample(points, args.max_points)

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "region_id"])
        for x, y in points:
            writer.writerow([x, y, "", args.region_id])

    print(f"Wrote {len(points)} points to {output_path}")


if __name__ == "__main__":
    main()
