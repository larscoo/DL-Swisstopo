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
    # Approximate bounding box around Basel-Stadt canton (EPSG:2056).
    "basel_stadt": (2609000.0, 1265500.0, 2614000.0, 1270500.0),
    # Approximate core area around central Basel (EPSG:2056).
    "basel_stadt_core": (2611100.0, 1267000.0, 2612900.0, 1269000.0),
}

def frange(start: float, stop: float, step: float):
    value = start
    while value <= stop:
        yield value
        value += step


def choose_non_overlapping_blocks(
    nx: int, ny: int, block_size: int, num_blocks: int, seed: int
) -> list[tuple[int, int]]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if block_size > nx or block_size > ny:
        raise ValueError("block_size does not fit in the grid dimensions")

    rng = random.Random(seed)
    anchors = [(ax, ay) for ax in range(nx - block_size + 1) for ay in range(ny - block_size + 1)]
    rng.shuffle(anchors)

    used_cells: set[tuple[int, int]] = set()
    selected: list[tuple[int, int]] = []

    for ax, ay in anchors:
        cells = [(ax + dx, ay + dy) for dx in range(block_size) for dy in range(block_size)]
        if any(cell in used_cells for cell in cells):
            continue
        selected.append((ax, ay))
        used_cells.update(cells)
        if len(selected) >= num_blocks:
            break

    if len(selected) < num_blocks:
        raise ValueError(
            f"Could not place {num_blocks} non-overlapping blocks of size {block_size}x{block_size}. "
            f"Only found {len(selected)}."
        )

    return selected

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grid points CSV.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()))
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--step", type=float, default=25.0, help="Grid spacing in meters.")
    parser.add_argument(
        "--sampling",
        choices=["random", "blocks"],
        default="random",
        help="Point sampling strategy.",
    )
    parser.add_argument("--max-points", type=int, default=300, help="Random sample cap.")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=60,
        help="Number of contiguous blocks (only for --sampling blocks).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4,
        help="Block edge length in grid cells (only for --sampling blocks).",
    )
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

    x_values = list(frange(xmin, xmax, args.step))
    y_values = list(frange(ymin, ymax, args.step))
    points: list[tuple[float, float, str]] = []

    if args.sampling == "random":
        all_points = [(x, y) for x in x_values for y in y_values]
        if args.max_points and len(all_points) > args.max_points:
            random.seed(args.seed)
            all_points = random.sample(all_points, args.max_points)
        points = [(x, y, "") for x, y in all_points]
    else:
        block_anchors = choose_non_overlapping_blocks(
            nx=len(x_values),
            ny=len(y_values),
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            seed=args.seed,
        )
        for block_no, (ax, ay) in enumerate(block_anchors, start=1):
            block_id = f"block_{block_no:04d}"
            for dx in range(args.block_size):
                for dy in range(args.block_size):
                    x = x_values[ax + dx]
                    y = y_values[ay + dy]
                    points.append((x, y, block_id))

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "region_id", "block_id"])
        for x, y, block_id in points:
            writer.writerow([x, y, "", args.region_id, block_id])

    print(f"Wrote {len(points)} points to {output_path}")


if __name__ == "__main__":
    main()
