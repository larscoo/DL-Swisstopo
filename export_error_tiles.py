#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy FP/FN images from an analysis folder into review directories.")
    parser.add_argument(
        "--analysis-dir",
        default="artifacts/analysis/sampler_only_effb0",
        help="Analysis directory produced by analyze_errors.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional export directory. Defaults to <analysis-dir>/tiles.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to export per error type.",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten filenames and prefix them with the score instead of recreating subdirectories.",
    )
    return parser.parse_args()


def resolve_source_path(project_root: Path, row: dict[str, str]) -> Path:
    for key in ("resolved_image_path", "image_path"):
        raw_path = str(row.get(key, "")).strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
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
    raise FileNotFoundError(f"Could not resolve source path from row: {row}")


def sanitized_score(score_value: str) -> str:
    return str(score_value or "0").replace(".", "_")


def copy_rows(csv_path: Path, destination_dir: Path, project_root: Path, limit: int | None, flatten: bool) -> int:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    copied = 0

    for row in rows:
        if limit is not None and copied >= limit:
            break

        source_path = resolve_source_path(project_root, row)
        destination_dir.mkdir(parents=True, exist_ok=True)

        if flatten:
            target_name = f"{sanitized_score(row.get('score', '0'))}_{source_path.name}"
            target_path = destination_dir / target_name
        else:
            target_path = destination_dir / source_path.name

        shutil.copy2(source_path, target_path)
        copied += 1

    return copied


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.is_absolute():
        analysis_dir = (project_root / analysis_dir).resolve()

    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (analysis_dir / "tiles")
    )
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    fp_csv = analysis_dir / "false_positives.csv"
    fn_csv = analysis_dir / "false_negatives.csv"
    if not fp_csv.exists() or not fn_csv.exists():
        raise FileNotFoundError(
            f"Expected false_positives.csv and false_negatives.csv in {analysis_dir}"
        )

    fp_count = copy_rows(fp_csv, output_dir / "false_positives", project_root, args.limit, args.flatten)
    fn_count = copy_rows(fn_csv, output_dir / "false_negatives", project_root, args.limit, args.flatten)

    print(f"Copied {fp_count} false positives to {output_dir / 'false_positives'}")
    print(f"Copied {fn_count} false negatives to {output_dir / 'false_negatives'}")


if __name__ == "__main__":
    main()
