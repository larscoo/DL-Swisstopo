from __future__ import annotations

from pathlib import Path


PREFERRED_DIRS = (
    ("runs/server", 0),
    ("runs/server/completed", 1),
    ("runs/local", 2),
)
DEPRIORITIZED_DIRS = (
    ("runs/server/seeds", 10),
    ("runs/server/incomplete", 20),
    ("experiments", 30),
)
DEFAULT_CHECKPOINT_POINTER = Path("artifacts/default_checkpoint.txt")
DEFAULT_THRESHOLD_POINTER = Path("artifacts/default_threshold.txt")


def _collect_pt_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.pt") if path.is_file())


def _collect_direct_run_pt_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    paths: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        paths.extend(sorted(path for path in child.rglob("*.pt") if path.is_file()))
    return paths


def _checkpoint_label(artifacts_dir: Path, checkpoint_path: Path) -> str:
    relative_path = checkpoint_path.relative_to(artifacts_dir)
    if checkpoint_path.name == "best_model.pt":
        return relative_path.parent.as_posix()
    return relative_path.as_posix()


def _read_preferred_checkpoint(base_dir: Path) -> Path | None:
    pointer_path = base_dir / DEFAULT_CHECKPOINT_POINTER
    if not pointer_path.exists():
        return None

    raw_value = pointer_path.read_text(encoding="utf-8").strip()
    if not raw_value:
        return None

    candidate = Path(raw_value)
    resolved = candidate if candidate.is_absolute() else (base_dir / candidate)
    resolved = resolved.resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Default checkpoint pointer references a path outside the workspace: {raw_value}"
        ) from exc
    if not resolved.exists():
        raise FileNotFoundError(
            f"Default checkpoint pointer references a missing file: {resolved}"
        )
    return resolved


def _read_preferred_threshold(base_dir: Path) -> float | None:
    pointer_path = base_dir / DEFAULT_THRESHOLD_POINTER
    if not pointer_path.exists():
        return None

    raw_value = pointer_path.read_text(encoding="utf-8").strip()
    if not raw_value:
        return None

    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Default threshold pointer contains an invalid float: {raw_value!r}") from exc

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Default threshold must be between 0.0 and 1.0, got {threshold}.")
    return threshold


def list_available_checkpoints(
    base_dir: Path,
    *,
    include_incomplete: bool = False,
    include_experiments: bool = False,
) -> list[dict[str, str]]:
    artifacts_dir = base_dir / "artifacts"
    ranked: list[tuple[int, float, Path]] = []
    seen: set[Path] = set()
    preferred_checkpoint = _read_preferred_checkpoint(base_dir)

    for relative_dir, priority in PREFERRED_DIRS:
        collector = _collect_direct_run_pt_files if relative_dir in {"runs/server", "runs/local"} else _collect_pt_files
        for path in collector(artifacts_dir / relative_dir):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            ranked.append((priority, -path.stat().st_mtime, path))

    if include_incomplete:
        for relative_dir, priority in DEPRIORITIZED_DIRS[1:2]:
            for path in _collect_pt_files(artifacts_dir / relative_dir):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                ranked.append((priority, -path.stat().st_mtime, path))

    if include_experiments:
        for relative_dir, priority in (DEPRIORITIZED_DIRS[:1] + DEPRIORITIZED_DIRS[2:]):
            for path in _collect_pt_files(artifacts_dir / relative_dir):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                ranked.append((priority, -path.stat().st_mtime, path))

    if not ranked:
        for path in _collect_pt_files(artifacts_dir):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            ranked.append((99, -path.stat().st_mtime, path))

    ranked.sort(key=lambda item: (item[0], item[1], item[2].as_posix()))
    if preferred_checkpoint is not None:
        preferred_checkpoint = preferred_checkpoint.resolve()
        ranked.sort(
            key=lambda item: (
                0 if item[2].resolve() == preferred_checkpoint else 1,
                item[0],
                item[1],
                item[2].as_posix(),
            )
        )
    checkpoints: list[dict[str, str]] = []
    for _, _, path in ranked:
        rel_from_base = path.relative_to(base_dir).as_posix()
        checkpoints.append(
            {
                "path": rel_from_base,
                "label": _checkpoint_label(artifacts_dir, path),
            }
        )
    return checkpoints


def resolve_default_checkpoint_path(base_dir: Path) -> Path:
    preferred_checkpoint = _read_preferred_checkpoint(base_dir)
    if preferred_checkpoint is not None:
        return preferred_checkpoint

    checkpoints = list_available_checkpoints(base_dir)
    if not checkpoints:
        raise FileNotFoundError("No model checkpoints found under artifacts/.")
    return (base_dir / checkpoints[0]["path"]).resolve()


def resolve_inference_threshold(
    base_dir: Path,
    checkpoint_path: Path,
    checkpoint_threshold: float,
) -> float:
    preferred_checkpoint = _read_preferred_checkpoint(base_dir)
    preferred_threshold = _read_preferred_threshold(base_dir)
    if preferred_checkpoint is None or preferred_threshold is None:
        return checkpoint_threshold

    if checkpoint_path.resolve() == preferred_checkpoint.resolve():
        return preferred_threshold
    return checkpoint_threshold
