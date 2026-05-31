#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from artifact_utils import resolve_default_checkpoint_path
from train import build_model, default_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap for one input image.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--input", default=None, help="Optional input image path.")
    parser.add_argument("--layer", default=None, help="Optional Conv2d layer name to use for Grad-CAM.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to plots/grad_cam/<run-name>/.",
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


def normalize_map(cam: torch.Tensor) -> torch.Tensor:
    cam = cam - cam.min()
    maximum = cam.max()
    if torch.isclose(maximum, torch.tensor(0.0)):
        return torch.zeros_like(cam)
    return cam / maximum


def compute_grad_cam(model: nn.Module, target_module: nn.Module, image_tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        captured["activations"] = output

    def backward_hook(_module, _grad_input, grad_output):
        captured["gradients"] = grad_output[0]

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor.unsqueeze(0))
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        score = logits.squeeze()
        score.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    activations = captured.get("activations")
    gradients = captured.get("gradients")
    if activations is None or gradients is None:
        raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

    activations = activations.detach().cpu()[0]
    gradients = gradients.detach().cpu()[0]
    weights = gradients.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=0))
    cam = normalize_map(cam)
    probability = float(torch.sigmoid(score.detach()).cpu().item())
    return cam, probability


def resize_cam_to_image(cam: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    cam_uint8 = (cam.numpy() * 255).astype(np.uint8)
    resized = Image.fromarray(cam_uint8, mode="L").resize(size, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_grad_cam_figure(
    output_path: Path,
    original_image: Image.Image,
    cam: np.ndarray,
    layer_name: str,
    checkpoint_name: str,
    probability: float,
) -> None:
    original_array = np.asarray(original_image)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_array)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(original_array)
    axes[2].imshow(cam, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"{checkpoint_name} | layer: {layer_name} | p={probability:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    progress(1, 5, "Resolving checkpoint and input image")
    checkpoint_path = resolve_path(project_root, args.checkpoint, resolve_default_checkpoint_path(project_root))
    input_path = resolve_path(project_root, args.input, default_input_image(project_root))
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

    progress(3, 5, "Computing Grad-CAM")
    original_image = Image.open(input_path).convert("RGB")
    image_tensor = eval_transform(original_image)
    cam, probability = compute_grad_cam(model, target_module, image_tensor)
    cam_resized = resize_cam_to_image(cam, original_image.size)

    progress(4, 5, "Preparing output path")
    run_name = checkpoint_path.parent.name
    output_dir = resolve_path(project_root, args.output_dir, project_root / "plots" / "grad_cam" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_{layer_name.replace('.', '_')}.png"

    progress(5, 5, f"Saving visualization to {output_path}")
    checkpoint_name = str(model_info.get("model_name") or run_name)
    save_grad_cam_figure(output_path, original_image, cam_resized, layer_name, checkpoint_name, probability)
    print(f"Saved Grad-CAM visualization to {output_path}", flush=True)


if __name__ == "__main__":
    main()
