#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn

from artifact_utils import resolve_default_checkpoint_path
from train import build_model, default_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize feature maps for a checkpoint and input image.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to the preferred checkpoint.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional input image path. Defaults to the first image found in data/y or data/n.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional module name to hook. Defaults to the last Conv2d layer in the model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Number of feature maps to render. Defaults to 16.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to plots/feature_maps/<run-name>/.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print available Conv2d layer names and exit.",
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


def default_input_image(project_root: Path) -> Path:
    for folder_name in ("data/y", "data/n"):
        folder = project_root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                return path.resolve()
    raise FileNotFoundError("No input image found under data/y or data/n.")


def load_checkpoint_model(checkpoint_path: Path) -> tuple[nn.Module, dict[str, object], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint.get("args", {})
    model_info = checkpoint.get("model_info", {})
    model_name = str(model_info.get("model_name") or model_args.get("model") or "simple_cnn")
    image_size = int(model_args.get("image_size", 224))

    model, _ = build_model(model_name, "off")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_info, image_size


def conv_layer_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]


def resolve_target_layer(model: nn.Module, layer_name: str | None) -> tuple[str, nn.Module]:
    convs = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if not convs:
        raise ValueError("Model does not contain any Conv2d layers.")

    if layer_name is None:
        return convs[-1]

    for name, module in convs:
        if name == layer_name:
            return name, module
    raise ValueError(f"Layer not found or not Conv2d: {layer_name}")


def capture_feature_maps(model: nn.Module, target_module: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output.detach().cpu()

    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(image_tensor.unsqueeze(0))
    finally:
        handle.remove()

    if "activation" not in captured:
        raise RuntimeError("No activation was captured from the selected layer.")
    activation = captured["activation"]
    if activation.ndim != 4:
        raise RuntimeError(f"Expected 4D activation tensor, got shape {tuple(activation.shape)}")
    return activation[0]


def rank_feature_maps(feature_maps: torch.Tensor) -> list[int]:
    flattened = feature_maps.view(feature_maps.shape[0], -1)
    scores = flattened.std(dim=1)
    ranked = torch.argsort(scores, descending=True)
    return ranked.tolist()


def normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    minimum = torch.min(feature_map)
    maximum = torch.max(feature_map)
    if torch.isclose(maximum, minimum):
        return torch.zeros_like(feature_map)
    return (feature_map - minimum) / (maximum - minimum)


def save_visualization(
    output_path: Path,
    original_image: Image.Image,
    feature_maps: torch.Tensor,
    selected_indices: list[int],
    layer_name: str,
    checkpoint_name: str,
) -> None:
    count = len(selected_indices)
    columns = min(4, max(1, count))
    rows = math.ceil((count + 1) / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    axes_list[0].imshow(original_image)
    axes_list[0].set_title("Input")
    axes_list[0].axis("off")

    for plot_index, channel_index in enumerate(selected_indices, start=1):
        ax = axes_list[plot_index]
        normalized = normalize_feature_map(feature_maps[channel_index]).numpy()
        ax.imshow(normalized, cmap="viridis")
        ax.set_title(f"ch {channel_index}")
        ax.axis("off")

    for ax in axes_list[count + 1 :]:
        ax.axis("off")

    fig.suptitle(f"{checkpoint_name} | layer: {layer_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    progress(1, 5, "Resolving checkpoint and input image")
    checkpoint_path = resolve_path(
        project_root,
        args.checkpoint,
        resolve_default_checkpoint_path(project_root),
    )
    input_path = resolve_path(
        project_root,
        args.input,
        default_input_image(project_root),
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    progress(2, 5, "Loading model")
    model, model_info, image_size = load_checkpoint_model(checkpoint_path)
    available_layers = conv_layer_names(model)
    if args.list_layers:
        print("\n".join(available_layers))
        return

    layer_name, target_module = resolve_target_layer(model, args.layer)
    _, eval_transform = default_transforms(image_size)

    progress(3, 5, "Running forward pass and capturing activations")
    original_image = Image.open(input_path).convert("RGB")
    image_tensor = eval_transform(original_image)
    feature_maps = capture_feature_maps(model, target_module, image_tensor)

    progress(4, 5, "Selecting informative feature maps")
    ranked_indices = rank_feature_maps(feature_maps)
    top_k = max(1, min(args.top_k, feature_maps.shape[0]))
    selected_indices = ranked_indices[:top_k]

    run_name = checkpoint_path.parent.name
    output_dir = resolve_path(
        project_root,
        args.output_dir,
        project_root / "plots" / "feature_maps" / run_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_{layer_name.replace('.', '_')}.png"

    progress(5, 5, f"Saving visualization to {output_path}")
    checkpoint_name = str(model_info.get("model_name") or run_name)
    save_visualization(output_path, original_image, feature_maps, selected_indices, layer_name, checkpoint_name)
    print(f"Saved feature map visualization to {output_path}", flush=True)


if __name__ == "__main__":
    main()
