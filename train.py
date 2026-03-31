#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    region_id: str


class TileDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
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
    parser = argparse.ArgumentParser(description="Train a baseline crosswalk classifier from labels.csv.")
    parser.add_argument("--csv-path", default="labels.csv", help="Path to labels CSV.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for checkpoints and metrics.")
    parser.add_argument(
        "--model",
        choices=["auto", "resnet18", "simple_cnn"],
        default="auto",
        help="Model architecture. 'auto' prefers resnet18 when torchvision is available.",
    )
    parser.add_argument(
        "--train-regions",
        nargs="*",
        default=None,
        help="Explicit region_id values for training.",
    )
    parser.add_argument(
        "--val-regions",
        nargs="*",
        default=None,
        help="Explicit region_id values for validation.",
    )
    parser.add_argument(
        "--test-regions",
        nargs="*",
        default=None,
        help="Explicit region_id values for testing.",
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


def load_samples(csv_path: Path) -> list[Sample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    samples: list[Sample] = []
    skipped_missing = 0
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

            samples.append(
                Sample(
                    image_path=image_path,
                    label=label,
                    region_id=str(row.get("region_id", "")).strip() or "region_unknown",
                )
            )

    if not samples:
        raise ValueError("No labeled samples found. Check labels.csv and image paths.")
    if skipped_missing:
        print(f"Skipped {skipped_missing} labeled rows because image files were missing.")
    return samples


def build_region_split(
    samples: list[Sample],
    train_regions: list[str] | None,
    val_regions: list[str] | None,
    test_regions: list[str] | None,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample], dict[str, list[str]]]:
    by_region: dict[str, list[Sample]] = {}
    for sample in samples:
        by_region.setdefault(sample.region_id, []).append(sample)

    all_regions = sorted(by_region)
    if train_regions is not None or val_regions is not None or test_regions is not None:
        train_regions = train_regions or []
        val_regions = val_regions or []
        test_regions = test_regions or []
    elif len(all_regions) >= 3:
        train_regions = all_regions[:-2]
        val_regions = [all_regions[-2]]
        test_regions = [all_regions[-1]]
    elif len(all_regions) == 2:
        train_regions = [all_regions[0]]
        val_regions = []
        test_regions = [all_regions[1]]
    else:
        shuffled = list(samples)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = max(int(n_total * 0.7), 1)
        n_val = max(int(n_total * 0.15), 1) if n_total >= 3 else 0
        train_split = shuffled[:n_train]
        val_split = shuffled[n_train : n_train + n_val]
        test_split = shuffled[n_train + n_val :]
        split_info = {
            "train_regions": ["random_split"],
            "val_regions": ["random_split"] if val_split else [],
            "test_regions": ["random_split"] if test_split else [],
        }
        print("Only one region with labels found. Falling back to a random split.")
        return train_split, val_split, test_split, split_info

    declared = set(train_regions) | set(val_regions) | set(test_regions)
    unknown = declared.difference(all_regions)
    if unknown:
        raise ValueError(f"Unknown regions in split arguments: {sorted(unknown)}")

    overlap = (
        (set(train_regions) & set(val_regions))
        | (set(train_regions) & set(test_regions))
        | (set(val_regions) & set(test_regions))
    )
    if overlap:
        raise ValueError(f"Regions may only appear in one split: {sorted(overlap)}")

    train_split = [sample for sample in samples if sample.region_id in set(train_regions)]
    val_split = [sample for sample in samples if sample.region_id in set(val_regions)]
    test_split = [sample for sample in samples if sample.region_id in set(test_regions)]
    split_info = {
        "train_regions": train_regions,
        "val_regions": val_regions,
        "test_regions": test_regions,
    }
    return train_split, val_split, test_split, split_info


def default_transforms(image_size: int) -> tuple[object, object]:
    if HAS_TORCHVISION:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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


def build_model(name: str) -> nn.Module:
    if name == "auto":
        name = "resnet18" if HAS_TORCHVISION else "simple_cnn"

    if name == "resnet18":
        if not HAS_TORCHVISION:
            raise RuntimeError("torchvision is not installed; resnet18 is unavailable.")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    return SimpleCNN()


def make_loader(samples: list[Sample], transform, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = TileDataset(samples, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int()
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
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[str, float]]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    if not all_logits:
        return 0.0, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    loss = running_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(logits, targets)
    return loss, metrics


def describe_split(name: str, samples: list[Sample]) -> str:
    counts = Counter(sample.label for sample in samples)
    return (
        f"{name}: n={len(samples)} | "
        f"neg={counts.get(0, 0)} | pos={counts.get(1, 0)} | "
        f"regions={sorted({sample.region_id for sample in samples})}"
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(csv_path)
    train_samples, val_samples, test_samples, split_info = build_region_split(
        samples=samples,
        train_regions=args.train_regions,
        val_regions=args.val_regions,
        test_regions=args.test_regions,
        seed=args.seed,
    )

    if not train_samples:
        raise ValueError("Training split is empty. Adjust the region split.")

    train_transform, eval_transform = default_transforms(args.image_size)
    train_loader = make_loader(train_samples, train_transform, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_samples, eval_transform, args.batch_size, args.num_workers, shuffle=False)
    test_loader = make_loader(test_samples, eval_transform, args.batch_size, args.num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)

    train_counts = Counter(sample.label for sample in train_samples)
    pos = train_counts.get(1, 0)
    neg = train_counts.get(0, 0)
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(describe_split("Train", train_samples))
    print(describe_split("Val", val_samples))
    print(describe_split("Test", test_samples))
    print(f"Model: {model.__class__.__name__} | Device: {device.type} | pos_weight={pos_weight.item():.2f}")

    best_val_f1 = -1.0
    best_state_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        if val_samples:
            val_loss, val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None)
        else:
            val_loss, val_metrics = 0.0, train_metrics

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.3f} | "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.3f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model.__class__.__name__,
                    "args": vars(args),
                    "split_info": split_info,
                },
                best_state_path,
            )

    if best_state_path.exists():
        checkpoint = torch.load(best_state_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if test_samples:
        test_loss, test_metrics = run_epoch(model, test_loader, device, criterion, optimizer=None)
    else:
        test_loss, test_metrics = 0.0, {}
    metrics_payload = {
        "split_info": split_info,
        "history": history,
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
            f"f1={test_metrics['f1']:.3f}"
        )
    print(f"Saved checkpoint to {best_state_path}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
