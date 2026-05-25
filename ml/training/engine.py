"""Training and validation loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import MetricResult, MetricTracker
from training.model import freeze_backbone, unfreeze_backbone

if TYPE_CHECKING:
    from training.config import TrainConfig


@dataclass(frozen=True, slots=True)
class EpochResult:
    train: MetricResult
    val: MetricResult
    learning_rate: float


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> MetricResult:
    model.train()
    tracker = MetricTracker()

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        tracker.update_logits(float(loss.item()), logits.detach().cpu(), targets.detach().cpu())

    return tracker.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> MetricResult:
    model.eval()
    tracker = MetricTracker()

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        tracker.update_logits(float(loss.item()), logits.cpu(), targets.cpu())

    return tracker.compute()


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    class_weights: torch.Tensor | None = None,
) -> tuple[nn.Module, MetricResult]:
    device = _device()
    model = model.to(device)

    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(device="cuda") if config.use_amp and device.type == "cuda" else None

    if config.freeze_backbone_epochs > 0:
        freeze_backbone(model, config.backbone)

    best_val = MetricResult(
        accuracy=0.0,
        top3_accuracy=0.0,
        f1_macro=0.0,
        f1_weighted=0.0,
        loss=float("inf"),
    )
    best_state = None

    for epoch in range(config.epochs):
        if epoch == config.freeze_backbone_epochs:
            unfreeze_backbone(model, config.backbone)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate * 0.5,
                weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.epochs - epoch, 1),
            )

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config.use_amp
        )
        val_metrics = validate(model, val_loader, criterion, device, config.use_amp)
        scheduler.step()

        print(
            f"epoch {epoch + 1}/{config.epochs} | "
            f"train acc={train_metrics.accuracy:.4f} top3={train_metrics.top3_accuracy:.4f} | "
            f"val acc={val_metrics.accuracy:.4f} top3={val_metrics.top3_accuracy:.4f} "
            f"f1={val_metrics.f1_macro:.4f}"
        )

        if val_metrics.f1_macro >= best_val.f1_macro:
            best_val = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val
