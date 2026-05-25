"""Augmentations and normalization."""

from __future__ import annotations

from typing import Literal

import torch
from torchvision import transforms as T

from training.config import TrainConfig
from training.labels import CLASS_NAMES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    config: TrainConfig,
    split: Literal["train", "val"],
) -> T.Compose:
    resize = T.Resize((config.image_size, config.image_size), antialias=True)

    if split == "train":
        return T.Compose(
            [
                resize,
                T.RandomApply(
                    [T.ColorJitter(brightness=config.aug_color_jitter, contrast=config.aug_color_jitter)],
                    p=0.8,
                ),
                T.RandomAffine(
                    degrees=config.aug_rotation,
                    translate=(config.aug_translate, config.aug_translate),
                    scale=config.aug_scale,
                    shear=4,
                ),
                T.RandomPerspective(distortion_scale=config.aug_perspective, p=0.35),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=config.aug_blur_prob),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                T.RandomErasing(p=0.1, scale=(0.02, 0.08)),
            ]
        )

    return T.Compose(
        [
            resize,
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def inference_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size), antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def class_weights_from_counts(counts: dict[str, int]) -> torch.Tensor:
    """Inverse-frequency weights for imbalanced splits."""
    total = sum(counts.get(name, 0) for name in CLASS_NAMES)
    if total == 0:
        return torch.ones(len(CLASS_NAMES))

    weights: list[float] = []
    for name in CLASS_NAMES:
        count = max(counts.get(name, 0), 1)
        weights.append(total / (len(CLASS_NAMES) * count))
    return torch.tensor(weights, dtype=torch.float32)
