"""CLI entry point for training."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from training.config import TrainConfig
from training.dataset import create_dataloaders, loss_weights_from_stats
from training.engine import run_training
from training.export import export_onnx, save_checkpoint
from training.model import create_model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train chess square piece classifier")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backbone", choices=["mobilenet_v3_small", "efficientnet_b0"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-onnx", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    overrides: dict = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.backbone:
        overrides["backbone"] = args.backbone
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["learning_rate"] = args.lr
    if args.no_amp:
        overrides["use_amp"] = False
    if args.no_onnx:
        overrides["export_onnx"] = False

    config = TrainConfig(**overrides)
    _seed_everything(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "best.pt"

    print(f"Loading data from {config.data_dir}")
    train_loader, val_loader, train_stats, val_stats = create_dataloaders(config)
    print(f"Train samples: {train_stats.num_samples} | Val samples: {val_stats.num_samples}")
    print(f"Class counts (train): {train_stats.class_counts}")

    model = create_model(config, pretrained=True)
    class_weights = loss_weights_from_stats(train_stats)

    model, best_metrics = run_training(
        model,
        train_loader,
        val_loader,
        config,
        class_weights=class_weights,
    )

    metrics_dict = {
        "accuracy": best_metrics.accuracy,
        "top3_accuracy": best_metrics.top3_accuracy,
        "f1_macro": best_metrics.f1_macro,
        "f1_weighted": best_metrics.f1_weighted,
        "loss": best_metrics.loss,
        "confusion": best_metrics.confusion,
    }
    save_checkpoint(checkpoint_path, model, config, metrics_dict)
    print(f"Saved checkpoint -> {checkpoint_path}")
    print(
        f"Best val: acc={best_metrics.accuracy:.4f} "
        f"top3={best_metrics.top3_accuracy:.4f} f1_macro={best_metrics.f1_macro:.4f}"
    )

    if config.export_onnx:
        onnx_path = config.output_dir / "piece_classifier.onnx"
        export_onnx(model, config, onnx_path)
        print(f"Exported ONNX -> {onnx_path}")


if __name__ == "__main__":
    main()
