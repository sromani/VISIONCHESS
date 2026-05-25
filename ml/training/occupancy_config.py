"""Binary occupancy model training configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ML_ROOT = Path(__file__).resolve().parents[1]


class OccupancyTrainConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VC_OCC_", extra="ignore")

    data_dir: Path = ML_ROOT / "data" / "occupancy"
    output_dir: Path = ML_ROOT / "models" / "occupancy"
    image_size: int = 64
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 3e-4
    occupied_threshold: float = 0.5
    export_onnx: bool = True

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.data_dir / "val"
