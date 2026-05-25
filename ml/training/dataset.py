"""Dataset and DataLoader construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import Compose

from training.config import TrainConfig
from training.labels import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
from training.transforms import build_transforms, class_weights_from_counts

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True, slots=True)
class DatasetStats:
    class_counts: dict[str, int]
    num_samples: int


class SquarePieceDataset(Dataset[tuple[torch.Tensor, int]]):
    """Loads ``root/<class_name>/*`` with labels mapped to ``CLASS_NAMES`` order."""

    def __init__(self, root: Path, transform: Compose) -> None:
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        missing = [name for name in CLASS_NAMES if not (self.root / name).is_dir()]
        if missing:
            msg = f"Missing class folders in {self.root}: {missing}"
            raise FileNotFoundError(msg)

        for class_name in CLASS_NAMES:
            label = CLASS_TO_IDX[class_name]
            class_dir = self.root / class_name
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

        if not self.samples:
            msg = f"No images found under {self.root}"
            raise FileNotFoundError(msg)

    @property
    def classes(self) -> list[str]:
        return list(CLASS_NAMES)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return dict(CLASS_TO_IDX)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return image, label

    def class_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return {name: counts.get(name, 0) for name in CLASS_NAMES}


def compute_stats(dataset: SquarePieceDataset) -> DatasetStats:
    counts = dataset.class_counts()
    return DatasetStats(class_counts=counts, num_samples=len(dataset))


def _build_sampler(dataset: SquarePieceDataset) -> WeightedRandomSampler | None:
    counts = dataset.class_counts()
    if min(counts.values()) == max(counts.values()):
        return None

    sample_weights: list[float] = []
    for _, label in dataset.samples:
        class_name = IDX_TO_CLASS[label]
        count = max(counts[class_name], 1)
        sample_weights.append(1.0 / count)

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DatasetStats, DatasetStats]:
    train_tf = build_transforms(config, "train")
    val_tf = build_transforms(config, "val")

    train_ds = SquarePieceDataset(config.train_dir, train_tf)
    val_ds = SquarePieceDataset(config.val_dir, val_tf)

    train_stats = compute_stats(train_ds)
    val_stats = compute_stats(val_ds)

    sampler = _build_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader, train_stats, val_stats


def loss_weights_from_stats(stats: DatasetStats) -> torch.Tensor:
    return class_weights_from_counts(stats.class_counts)
