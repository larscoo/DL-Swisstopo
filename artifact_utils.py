from __future__ import annotations

from pathlib import Path


PREFERRED_DIRS = (
    ("runs/server/completed", 0),
    ("runs/local", 1),
)
DEPRIORITIZED_DIRS = (
    ("runs/server/incomplete", 20),
    ("experiments", 30),
)


def _collect_pt_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.pt") if path.is_file())


def _checkpoint_label(artifacts_dir: Path, checkpoint_path: Path) -> str:
    relative_path = checkpoint_path.relative_to(artifacts_dir)
    if checkpoint_path.name == "best_model.pt":
        return relative_path.parent.as_posix()
    return relative_path.as_posix()


def list_available_checkpoints(
    base_dir: Path,
    *,
    include_incomplete: bool = False,
    include_experiments: bool = False,
) -> list[dict[str, str]]:
    artifacts_dir = base_dir / "artifacts"
    ranked: list[tuple[int, float, Path]] = []
    seen: set[Path] = set()

    for relative_dir, priority in PREFERRED_DIRS:
        for path in _collect_pt_files(artifacts_dir / relative_dir):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            ranked.append((priority, -path.stat().st_mtime, path))

    if include_incomplete:
        for relative_dir, priority in DEPRIORITIZED_DIRS[:1]:
            for path in _collect_pt_files(artifacts_dir / relative_dir):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                ranked.append((priority, -path.stat().st_mtime, path))

    if include_experiments:
        for relative_dir, priority in DEPRIORITIZED_DIRS[1:]:
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
    checkpoints = list_available_checkpoints(base_dir)
    if not checkpoints:
        raise FileNotFoundError("No model checkpoints found under artifacts/.")
    return (base_dir / checkpoints[0]["path"]).resolve()
