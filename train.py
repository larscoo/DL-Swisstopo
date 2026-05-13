#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from torchvision import models, transforms

    HAS_TORCHVISION = True
except ImportError:
    models = None
    transforms = None
    HAS_TORCHVISION = False


LABEL_MAP = {
    "0": 0,
    "1": 1,
    "no_crosswalk": 0,
    "crosswalk": 1,
}
IGNORED_LABELS = {"", "ignore"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    x: int
    y: int
    split_group: str
    source: str


class TileDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[Sample], transform, label_transforms: dict[int, object] | None = None) -> None:
        self.samples = samples
        self.transform = transform
        self.label_transforms = label_transforms or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        transform = self.label_transforms.get(sample.label, self.transform)
        if transform is not None:
            image = transform(image)
        label = torch.tensor(sample.label, dtype=torch.float32)
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a crosswalk tile classifier from folder labels or a labels CSV.")
    parser.add_argument("--csv-path", default=None, help="Optional labels CSV such as data/labels.csv. If omitted, data/y and data/n are used.")
    parser.add_argument("--pos-dir", default="data/y", help="Directory with positive tiles.")
    parser.add_argument("--neg-dir", default="data/n", help="Directory with negative tiles.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for checkpoints, metrics and manifest. Defaults to a timestamped folder under artifacts/runs/local/.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. 'auto' prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "resnet18", "efficientnet_b0", "simple_cnn"],
        default="auto",
        help="Model architecture. 'auto' prefers EfficientNet-B0 when torchvision is available.",
    )
    parser.add_argument(
        "--pretrained",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use ImageNet weights when available. 'auto' enables them for torchvision models.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Target fraction of spatial groups for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Target fraction of spatial groups for validation.",
    )
    parser.add_argument(
        "--spatial-bin-size",
        type=int,
        default=1000,
        help="Spatial split bin size in EPSG:2056 meters.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Optional fixed threshold. If omitted, the best validation F1 threshold is used.",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use a weighted sampler so positives and negatives appear more evenly during training.",
    )
    parser.add_argument(
        "--imbalance-strategy",
        choices=["auto", "none", "pos_weight", "sampler", "both"],
        default="auto",
        help="How to handle class imbalance. 'auto' keeps the current behavior: pos_weight only, or both when --balanced-sampling is set.",
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=["standard", "class_aware"],
        default="standard",
        help="Training augmentation strategy. 'class_aware' applies stronger augmentations to positive samples.",
    )
    parser.add_argument(
        "--threshold-min-recall",
        type=float,
        default=0.0,
        help="Minimum validation recall required during threshold search. If no threshold satisfies it, best F1 is used.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.csv",
        help="Filename for the generated sample manifest inside output-dir.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(raw_label: str) -> int | None:
    label = str(raw_label or "").strip().lower()
    if label in IGNORED_LABELS:
        return None
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    raise ValueError(f"Unsupported label value: {raw_label!r}")


def parse_xy_from_stem(stem: str) -> tuple[int, int]:
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected filename stem '<x>_<y>', got: {stem!r}")
    return int(parts[0]), int(parts[1])


def spatial_group(x: int, y: int, bin_size: int) -> str:
    return f"{x // bin_size}_{y // bin_size}"


def load_samples_from_dirs(pos_dir: Path, neg_dir: Path, bin_size: int) -> list[Sample]:
    samples: list[Sample] = []
    skipped_hidden = 0

    for folder, label, source in ((pos_dir, 1, "folder_pos"), (neg_dir, 0, "folder_neg")):
        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {folder}")

        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                skipped_hidden += 1
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            x, y = parse_xy_from_stem(path.stem)
            samples.append(
                Sample(
                    image_path=path.resolve(),
                    label=label,
                    x=x,
                    y=y,
                    split_group=spatial_group(x, y, bin_size),
                    source=source,
                )
            )

    if not samples:
        raise ValueError("No labeled samples found in data/y and data/n.")
    if skipped_hidden:
        print(f"Skipped {skipped_hidden} hidden files while scanning data directories.")
    return samples


def load_samples_from_csv(csv_path: Path, bin_size: int) -> list[Sample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    samples: list[Sample] = []
    skipped_missing = 0
    skipped_invalid = 0

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = normalize_label(row.get("label", ""))
            if label is None:
                continue

            image_path = Path(str(row.get("image_path", "")).strip())
            if not image_path.is_absolute():
                image_path = (csv_path.parent / image_path).resolve()
            if not image_path.exists():
                skipped_missing += 1
                continue

            try:
                if row.get("x") and row.get("y"):
                    x = int(round(float(str(row["x"]).strip())))
                    y = int(round(float(str(row["y"]).strip())))
                else:
                    x, y = parse_xy_from_stem(image_path.stem)
            except ValueError:
                skipped_invalid += 1
                continue

            region_id = str(row.get("region_id", "")).strip()
            samples.append(
                Sample(
                    image_path=image_path,
                    label=label,
                    x=x,
                    y=y,
                    split_group=region_id or spatial_group(x, y, bin_size),
                    source="csv",
                )
            )

    if not samples:
        raise ValueError("No labeled samples found. Check the CSV path and image paths.")
    if skipped_missing:
        print(f"Skipped {skipped_missing} labeled rows because image files were missing.")
    if skipped_invalid:
        print(f"Skipped {skipped_invalid} labeled rows because coordinates could not be parsed.")
    return samples


def load_samples(args: argparse.Namespace) -> tuple[list[Sample], str]:
    if args.csv_path:
        return load_samples_from_csv(Path(args.csv_path), args.spatial_bin_size), "csv"
    return load_samples_from_dirs(Path(args.pos_dir), Path(args.neg_dir), args.spatial_bin_size), "folders"


def validate_images(samples: list[Sample]) -> list[Sample]:
    valid: list[Sample] = []
    skipped = 0
    for sample in samples:
        try:
            with Image.open(sample.image_path) as image:
                image.verify()
            valid.append(sample)
        except (OSError, UnidentifiedImageError):
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} unreadable image files during dataset validation.")
    if not valid:
        raise ValueError("No readable images found after validation.")
    return valid


def assign_groups_to_splits(
    group_stats: list[dict[str, int | str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected ratios with train_ratio > 0, val_ratio >= 0 and train_ratio + val_ratio < 1.")

    rng = random.Random(seed)
    shuffled = list(group_stats)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda item: (int(item["total"]), int(item["pos"])), reverse=True)

    total_items = sum(int(item["total"]) for item in shuffled)
    targets = {
        "train": total_items * train_ratio,
        "val": total_items * val_ratio,
        "test": total_items * max(1.0 - train_ratio - val_ratio, 0.0),
    }
    assigned_counts = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, str] = {}

    for item in shuffled:
        group = str(item["group"])
        group_total = int(item["total"])
        candidates = ["train", "val", "test"]
        candidates.sort(key=lambda split: (assigned_counts[split] / max(targets[split], 1.0), assigned_counts[split]))
        chosen = candidates[0]
        assignments[group] = chosen
        assigned_counts[chosen] += group_total

    # Avoid empty validation/test splits when enough groups exist.
    present_splits = {split for split in assignments.values()}
    if len(shuffled) >= 3:
        for split in ("val", "test"):
            if split in present_splits:
                continue
            donor = max(("train", "val", "test"), key=lambda s: assigned_counts[s])
            donor_groups = [item for item in shuffled if assignments[str(item["group"])] == donor]
            donor_groups.sort(key=lambda item: int(item["total"]))
            if donor_groups:
                assignments[str(donor_groups[0]["group"])] = split
                assigned_counts[donor] -= int(donor_groups[0]["total"])
                assigned_counts[split] += int(donor_groups[0]["total"])

    return assignments


def build_spatial_split(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample], dict[str, object]]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.split_group].append(sample)

    group_stats: list[dict[str, int | str]] = []
    for group, items in grouped.items():
        group_stats.append(
            {
                "group": group,
                "total": len(items),
                "pos": sum(sample.label for sample in items),
                "neg": len(items) - sum(sample.label for sample in items),
            }
        )

    assignments = assign_groups_to_splits(group_stats, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    train_samples = [sample for sample in samples if assignments[sample.split_group] == "train"]
    val_samples = [sample for sample in samples if assignments[sample.split_group] == "val"]
    test_samples = [sample for sample in samples if assignments[sample.split_group] == "test"]

    split_info = {
        "strategy": "spatial_group_split",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": max(1.0 - train_ratio - val_ratio, 0.0),
        "num_groups": len(grouped),
        "train_groups": sorted(group for group, split in assignments.items() if split == "train"),
        "val_groups": sorted(group for group, split in assignments.items() if split == "val"),
        "test_groups": sorted(group for group, split in assignments.items() if split == "test"),
    }
    return train_samples, val_samples, test_samples, split_info


def default_transforms(image_size: int) -> tuple[object, object]:
    if HAS_TORCHVISION:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.25),
                transforms.RandomApply([transforms.RandomRotation((180, 180))], p=0.25),
                transforms.RandomApply([transforms.RandomRotation((270, 270))], p=0.25),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        return train_transform, eval_transform

    def _to_tensor(image: Image.Image) -> torch.Tensor:
        image = image.resize((image_size, image_size))
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.view(image.size[1], image.size[0], len(image.getbands()))
        return data.permute(2, 0, 1).float() / 255.0

    return _to_tensor, _to_tensor


def build_train_transforms(image_size: int, augmentation_mode: str) -> tuple[object, dict[int, object] | None]:
    if augmentation_mode == "standard":
        train_transform, _ = default_transforms(image_size)
        return train_transform, None

    if not HAS_TORCHVISION:
        print("Warning: class-aware augmentation requires torchvision. Falling back to standard transforms.")
        train_transform, _ = default_transforms(image_size)
        return train_transform, None

    common = [transforms.Resize((image_size, image_size))]
    neg_transform = transforms.Compose(
        common
        + [
            transforms.RandomHorizontalFlip(p=0.15),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.01),
            transforms.ToTensor(),
        ]
    )
    pos_transform = transforms.Compose(
        common
        + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.2),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
        ]
    )
    return None, {0: neg_transform, 1: pos_transform}


def resolve_pretrained_flag(model_name: str, pretrained: str) -> bool:
    if pretrained == "on":
        return True
    if pretrained == "off":
        return False
    return model_name in {"resnet18", "efficientnet_b0"}


def build_model(name: str, pretrained: str) -> tuple[nn.Module, dict[str, object]]:
    if name == "auto":
        name = "efficientnet_b0" if HAS_TORCHVISION else "simple_cnn"

    if name == "simple_cnn":
        return SimpleCNN(), {"model_name": "simple_cnn", "pretrained_requested": False, "pretrained_loaded": False}

    if not HAS_TORCHVISION:
        raise RuntimeError("torchvision is not installed; torchvision backbones are unavailable.")

    use_pretrained = resolve_pretrained_flag(name, pretrained)
    model_info = {"model_name": name, "pretrained_requested": use_pretrained, "pretrained_loaded": False}

    if name == "resnet18":
        weights = None
        if use_pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = None
        try:
            model = models.resnet18(weights=weights)
            model_info["pretrained_loaded"] = weights is not None
        except Exception as exc:
            print(f"Warning: could not load pretrained ResNet18 weights ({exc}). Falling back to random init.")
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model, model_info

    if name == "efficientnet_b0":
        weights = None
        if use_pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            except AttributeError:
                weights = None
        try:
            model = models.efficientnet_b0(weights=weights)
            model_info["pretrained_loaded"] = weights is not None
        except Exception as exc:
            print(f"Warning: could not load pretrained EfficientNet-B0 weights ({exc}). Falling back to random init.")
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        return model, model_info

    raise ValueError(f"Unsupported model: {name}")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but Apple Metal is not available in this PyTorch install.")
        return torch.device("mps")

    return torch.device("cpu")


def resolve_imbalance_strategy(args: argparse.Namespace) -> str:
    if args.imbalance_strategy != "auto":
        return args.imbalance_strategy
    return "both" if args.balanced_sampling else "pos_weight"


def validate_threshold_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold_min_recall <= 1.0:
        raise ValueError("--threshold-min-recall must be between 0.0 and 1.0.")


def make_train_sampler(samples: list[Sample]) -> WeightedRandomSampler | None:
    counts = Counter(sample.label for sample in samples)
    if len(counts) < 2:
        return None
    sample_weights = [1.0 / counts[sample.label] for sample in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(samples), replacement=True)


def make_loader(
    samples: list[Sample],
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
    label_transforms: dict[int, object] | None = None,
) -> DataLoader:
    dataset = TileDataset(samples, transform, label_transforms=label_transforms)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def compute_pr_auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    if len(probs) == 0:
        return 0.0

    pairs = sorted(zip(probs.tolist(), targets.int().tolist()), key=lambda item: item[0], reverse=True)
    total_pos = sum(label for _, label in pairs)
    if total_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    prev_recall = 0.0
    auc = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / max(tp + fp, 1)
        auc += (recall - prev_recall) * precision
        prev_recall = recall
    return auc


def compute_metrics_at_threshold(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    probs = sigmoid_probs(logits)
    preds = (probs >= threshold).int()
    targets = targets.int()

    tp = int(((preds == 1) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())

    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / total
    pr_auc = compute_pr_auc(probs, targets)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(logits: torch.Tensor, targets: torch.Tensor, min_recall: float = 0.0) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_metrics_at_threshold(logits, targets, 0.5)
    best_valid_threshold: float | None = None
    best_valid_metrics: dict[str, float] | None = None
    for step in range(5, 96, 5):
        threshold = step / 100.0
        metrics = compute_metrics_at_threshold(logits, targets, threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics
        if metrics["recall"] >= min_recall:
            if best_valid_metrics is None:
                best_valid_threshold = threshold
                best_valid_metrics = metrics
                continue
            current_key = (metrics["f1"], metrics["precision"], metrics["recall"], threshold)
            best_key = (
                best_valid_metrics["f1"],
                best_valid_metrics["precision"],
                best_valid_metrics["recall"],
                float(best_valid_threshold),
            )
            if current_key > best_key:
                best_valid_threshold = threshold
                best_valid_metrics = metrics
    if best_valid_metrics is not None and best_valid_threshold is not None:
        return best_valid_threshold, best_valid_metrics
    return best_threshold, best_metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    is_training = optimizer is not None
    model.train(is_training)
    epoch_start = time.time()

    running_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            if not torch.isfinite(logits).all():
                raise ValueError(
                    f"Non-finite logits detected in batch {batch_idx}. "
                    "Training became numerically unstable."
                )
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss detected in batch {batch_idx}. "
                    "Training became numerically unstable."
                )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                for name, parameter in model.named_parameters():
                    if not torch.isfinite(parameter).all():
                        raise ValueError(
                            f"Non-finite weights detected after optimizer step in batch {batch_idx}: {name}. "
                            "Reduce the learning rate or switch to a more stable configuration."
                        )

        running_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    if not all_logits:
        return 0.0, torch.empty(0), torch.empty(0), time.time() - epoch_start

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    loss = running_loss / max(len(loader.dataset), 1)
    if not math.isfinite(loss):
        raise ValueError("Epoch loss is non-finite. Training became numerically unstable.")
    return loss, logits, targets, time.time() - epoch_start


def describe_split(name: str, samples: list[Sample]) -> str:
    counts = Counter(sample.label for sample in samples)
    groups = sorted({sample.split_group for sample in samples})
    return (
        f"{name}: n={len(samples)} | "
        f"neg={counts.get(0, 0)} | pos={counts.get(1, 0)} | "
        f"groups={len(groups)}"
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_manifest(path: Path, samples: list[Sample], split_assignments: dict[Path, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "x", "y", "split_group", "split", "source"])
        for sample in samples:
            writer.writerow(
                [
                    sample.image_path.as_posix(),
                    sample.label,
                    sample.x,
                    sample.y,
                    sample.split_group,
                    split_assignments[sample.image_path],
                    sample.source,
                ]
            )


def format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def format_wall_clock_from_now(offset_seconds: float) -> str:
    target = datetime.fromtimestamp(time.time() + max(offset_seconds, 0.0))
    return target.strftime("%H:%M")


def build_default_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return base_dir / "artifacts" / "runs" / "local" / f"{timestamp}-run"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_threshold_args(args)

    output_dir = Path(args.output_dir) if args.output_dir else build_default_output_dir(Path(__file__).resolve().parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, dataset_source = load_samples(args)
    samples = validate_images(samples)
    train_samples, val_samples, test_samples, split_info = build_spatial_split(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if not train_samples:
        raise ValueError("Training split is empty. Adjust the spatial split settings.")
    if not val_samples:
        print("Warning: validation split is empty. Threshold selection will fall back to 0.5.")

    split_assignments = {sample.image_path: "train" for sample in train_samples}
    split_assignments.update({sample.image_path: "val" for sample in val_samples})
    split_assignments.update({sample.image_path: "test" for sample in test_samples})
    save_manifest(output_dir / args.manifest_name, samples, split_assignments)

    train_transform, train_label_transforms = build_train_transforms(args.image_size, args.augmentation_mode)
    _, eval_transform = default_transforms(args.image_size)
    imbalance_strategy = resolve_imbalance_strategy(args)
    train_sampler = make_train_sampler(train_samples) if imbalance_strategy in {"sampler", "both"} else None
    device = resolve_device(args.device)
    train_loader = make_loader(
        train_samples,
        train_transform,
        args.batch_size,
        args.num_workers,
        shuffle=train_sampler is None,
        device=device,
        sampler=train_sampler,
        label_transforms=train_label_transforms,
    )
    val_loader = make_loader(
        val_samples,
        eval_transform,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        device=device,
    )
    test_loader = make_loader(
        test_samples,
        eval_transform,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        device=device,
    )

    model, model_info = build_model(args.model, args.pretrained)
    model = model.to(device)

    train_counts = Counter(sample.label for sample in train_samples)
    pos = train_counts.get(1, 0)
    neg = train_counts.get(0, 0)
    pos_weight_value = neg / max(pos, 1)
    pos_weight = None
    if imbalance_strategy in {"pos_weight", "both"}:
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(describe_split("Train", train_samples))
    print(describe_split("Val", val_samples))
    print(describe_split("Test", test_samples))
    print(
        f"Dataset source: {dataset_source} | Model: {model_info['model_name']} | "
        f"Device: {device.type} | pretrained_loaded={model_info['pretrained_loaded']} | "
        f"imbalance_strategy={imbalance_strategy} | augmentation_mode={args.augmentation_mode} | "
        f"pos_weight={'off' if pos_weight is None else f'{pos_weight.item():.2f}'} | "
        f"threshold_min_recall={args.threshold_min_recall:.2f}"
    )

    best_val_f1 = -1.0
    best_state_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    best_threshold = args.decision_threshold if args.decision_threshold is not None else 0.5
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_logits, train_targets, train_epoch_seconds = run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        train_metrics = compute_metrics_at_threshold(train_logits, train_targets, best_threshold)

        if val_samples:
            val_loss, val_logits, val_targets, val_epoch_seconds = run_epoch(
                model, val_loader, device, criterion, optimizer=None
            )
            if args.decision_threshold is None:
                epoch_threshold, val_metrics = find_best_threshold(
                    val_logits,
                    val_targets,
                    min_recall=args.threshold_min_recall,
                )
            else:
                epoch_threshold = args.decision_threshold
                val_metrics = compute_metrics_at_threshold(val_logits, val_targets, epoch_threshold)
        else:
            val_loss = 0.0
            val_epoch_seconds = 0.0
            epoch_threshold = best_threshold
            val_metrics = train_metrics

        epoch_seconds = train_epoch_seconds + val_epoch_seconds
        elapsed_seconds = time.time() - training_start
        avg_epoch_seconds = elapsed_seconds / epoch
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_seconds * remaining_epochs

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "train_pr_auc": train_metrics["pr_auc"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_pr_auc": val_metrics["pr_auc"],
                "threshold": epoch_threshold,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.3f} train_pr_auc={train_metrics['pr_auc']:.3f} | "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.3f} val_pr_auc={val_metrics['pr_auc']:.3f} | "
            f"thr={epoch_threshold:.2f} | "
            f"epoch_time={format_duration(epoch_seconds)} elapsed={format_duration(elapsed_seconds)} "
            f"eta={format_duration(eta_seconds)} finish~{format_wall_clock_from_now(eta_seconds)}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = epoch_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_info": model_info,
                    "args": vars(args),
                    "split_info": split_info,
                    "best_threshold": best_threshold,
                },
                best_state_path,
            )

    if best_state_path.exists():
        checkpoint = torch.load(best_state_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_threshold = float(checkpoint.get("best_threshold", best_threshold))

    if test_samples:
        test_loss, test_logits, test_targets, _ = run_epoch(model, test_loader, device, criterion, optimizer=None)
        test_metrics = compute_metrics_at_threshold(test_logits, test_targets, best_threshold)
    else:
        test_loss = 0.0
        test_metrics = {}

    metrics_payload = {
        "dataset_source": dataset_source,
        "model_info": model_info,
        "split_info": split_info,
        "training_config": {
            "augmentation_mode": args.augmentation_mode,
            "imbalance_strategy": imbalance_strategy,
            "threshold_min_recall": args.threshold_min_recall,
        },
        "history": history,
        "best_threshold": best_threshold,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "train_summary": describe_split("Train", train_samples),
        "val_summary": describe_split("Val", val_samples),
        "test_summary": describe_split("Test", test_samples),
    }
    save_json(output_dir / "metrics.json", metrics_payload)

    if test_samples:
        print(
            f"Test | loss={test_loss:.4f} acc={test_metrics['accuracy']:.3f} "
            f"precision={test_metrics['precision']:.3f} recall={test_metrics['recall']:.3f} "
            f"f1={test_metrics['f1']:.3f} pr_auc={test_metrics['pr_auc']:.3f} "
            f"thr={best_threshold:.2f}"
        )
    print(f"Saved checkpoint to {best_state_path}")
    print(f"Saved manifest to {output_dir / args.manifest_name}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
