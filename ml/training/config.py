"""Training configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ML_ROOT = Path(__file__).resolve().parents[1]


class TrainConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VC_ML_", env_file=".env", extra="ignore")

    data_dir: Path = ML_ROOT / "data" / "squares"
    output_dir: Path = ML_ROOT / "models" / "piece_classifier"

    backbone: Literal["mobilenet_v3_small", "efficientnet_b0"] = "mobilenet_v3_small"
    image_size: int = 64
    num_classes: int = 13

    epochs: int = 30
    batch_size: int = 128
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05

    freeze_backbone_epochs: int = 3
    use_amp: bool = True
    seed: int = 42

    # Augmentation
    aug_rotation: float = 8.0
    aug_translate: float = 0.06
    aug_scale: tuple[float, float] = (0.92, 1.08)
    aug_perspective: float = 0.08
    aug_color_jitter: float = 0.25
    aug_blur_prob: float = 0.15

    # Export
    export_onnx: bool = True
    onnx_opset: int = 18

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.data_dir / "val"

    checkpoint_path: Path = Field(default_factory=lambda: ML_ROOT / "models" / "piece_classifier" / "best.pt")
