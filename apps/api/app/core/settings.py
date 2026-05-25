"""Application configuration via environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "VisionChess API"
    app_version: str = "0.1.0"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    api_prefix: str = "/api/v1"
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "capacitor://localhost,ionic://localhost,http://localhost"
    )

    # Upload / storage
    storage_path: Path = Field(default=API_ROOT / "uploads")
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_image_types: str = "image/jpeg,image/png,image/webp"
    temp_file_ttl_hours: int = 24

    # Stockfish
    stockfish_path: str = "stockfish"
    stockfish_depth_default: int = 18
    stockfish_multipv_default: int = 3

    # Vision pipeline
    board_output_size: int = 800
    piece_classifier_path: str | None = None
    classification_margin_ratio: float = 0.10
    classification_use_stockfish_tiebreak: bool = False

    @field_validator("storage_path", mode="before")
    @classmethod
    def expand_storage_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_mime_types(self) -> frozenset[str]:
        return frozenset(t.strip() for t in self.allowed_image_types.split(",") if t.strip())

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
