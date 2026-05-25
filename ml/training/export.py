"""Checkpoint and ONNX export."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from training.config import TrainConfig
from training.labels import CLASS_NAMES, NUM_CLASSES


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "backbone": config.backbone,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "class_names": list(CLASS_NAMES),
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def export_onnx(
    model: nn.Module,
    config: TrainConfig,
    output_path: Path,
) -> Path:
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, config.image_size, config.image_size)
    device = next(model.parameters()).device
    dummy = dummy.to(device)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["square"],
        output_names=["logits"],
        dynamic_axes={"square": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=config.onnx_opset,
        do_constant_folding=True,
        dynamo=False,
    )

    metadata = {
        "backbone": config.backbone,
        "image_size": config.image_size,
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "input_name": "square",
        "output_name": "logits",
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path
