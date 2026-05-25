"""Train binary empty vs occupied model from piece dataset folders."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from training.occupancy_config import ML_ROOT, OccupancyTrainConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class BinaryOccupancyDataset(Dataset):
    def __init__(self, root: Path, transform: T.Compose) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = 0 if class_dir.name == "empty" else 1
            for path in class_dir.iterdir():
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label

    def label_counts(self) -> Counter[int]:
        return Counter(label for _, label in self.samples)


def build_model(*, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    return model


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for pred, target in zip(preds.view(-1), labels.view(-1), strict=True):
                p, t = int(pred), int(target)
                if t == 1 and p == 1:
                    tp += 1
                elif t == 0 and p == 1:
                    fp += 1
                elif t == 0 and p == 0:
                    tn += 1
                else:
                    fn += 1
    total = max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(fp + tn, 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary occupancy classifier")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    overrides: dict = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    config = OccupancyTrainConfig(**overrides)
    default_data = ML_ROOT / "data" / "occupancy"
    if not config.train_dir.exists() and default_data.exists():
        config = OccupancyTrainConfig(**{**overrides, "data_dir": default_data})

    train_transform = T.Compose(
        [
            T.Resize((config.image_size, config.image_size)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = BinaryOccupancyDataset(config.train_dir, train_transform)
    val_ds = BinaryOccupancyDataset(config.val_dir, val_transform)
    if len(train_ds) == 0 or len(val_ds) == 0:
        msg = f"No samples in {config.train_dir} or {config.val_dir}. Run generate_synthetic_dataset.py first."
        raise SystemExit(msg)

    counts = train_ds.label_counts()
    print(f"Train: {len(train_ds)} samples | empty={counts[0]} occupied={counts[1]}")
    print(f"Val:   {len(val_ds)} samples | counts={val_ds.label_counts()}")

    weights = [1.0 / max(counts[label], 1) for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(pretrained=not args.no_pretrained).to(device)
    class_weights = torch.tensor(
        [1.0 / max(counts[0], 1), 1.0 / max(counts[1], 1)],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.sum() * 2.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_metrics: dict[str, float] = {"f1": 0.0}

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        metrics = _evaluate(model, val_loader, device)
        scheduler.step()
        print(
            f"epoch {epoch + 1}/{config.epochs} loss={running_loss / max(n_batches, 1):.4f} "
            f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f} fpr={metrics['false_positive_rate']:.4f}"
        )

        if metrics["f1"] >= best_metrics.get("f1", 0.0):
            best_metrics = metrics
            torch.save(model.state_dict(), config.output_dir / "best.pt")

    if not (config.output_dir / "best.pt").exists():
        torch.save(model.state_dict(), config.output_dir / "best.pt")

    model.load_state_dict(torch.load(config.output_dir / "best.pt", map_location=device, weights_only=True))
    model.eval()

    if config.export_onnx:
        dummy = torch.randn(1, 3, config.image_size, config.image_size, device=device)
        onnx_path = config.output_dir / "occupancy.onnx"
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
        )
        meta = {
            "model_type": "occupancy_binary",
            "class_names": ["empty", "occupied"],
            "image_size": config.image_size,
            "occupied_threshold": config.occupied_threshold,
            "val_metrics": best_metrics,
        }
        (config.output_dir / "occupancy.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Exported {onnx_path}")
        print(f"Best val: acc={best_metrics['accuracy']:.4f} f1={best_metrics['f1']:.4f} fpr={best_metrics['false_positive_rate']:.4f}")


if __name__ == "__main__":
    main()
