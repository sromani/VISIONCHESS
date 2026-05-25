"""Transfer-learning backbones for square piece classification."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V3_Small_Weights

from training.config import TrainConfig


def build_mobilenet_v3_small(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(
    backbone: Literal["mobilenet_v3_small", "efficientnet_b0"],
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        return build_mobilenet_v3_small(num_classes, pretrained)
    if backbone == "efficientnet_b0":
        return build_efficientnet_b0(num_classes, pretrained)
    msg = f"Unknown backbone: {backbone}"
    raise ValueError(msg)


def freeze_backbone(model: nn.Module, backbone: str) -> None:
    if backbone == "mobilenet_v3_small":
        for param in model.features.parameters():
            param.requires_grad = False
        return

    if backbone == "efficientnet_b0":
        for param in model.features.parameters():
            param.requires_grad = False
        return

    msg = f"Unknown backbone: {backbone}"
    raise ValueError(msg)


def unfreeze_backbone(model: nn.Module, backbone: str) -> None:
    if backbone in {"mobilenet_v3_small", "efficientnet_b0"}:
        for param in model.features.parameters():
            param.requires_grad = True
        return
    msg = f"Unknown backbone: {backbone}"
    raise ValueError(msg)


def create_model(config: TrainConfig, pretrained: bool = True) -> nn.Module:
    return build_model(config.backbone, config.num_classes, pretrained=pretrained)
