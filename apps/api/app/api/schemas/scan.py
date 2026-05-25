"""Scan API schemas."""

from pydantic import BaseModel, Field

from app.api.schemas.common import PointSchema


class ScanResponse(BaseModel):
    id: str
    status: str
    fen: str | None = None
    fen_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warped_image_url: str | None = None
    original_image_url: str | None = None
    board_corners: list[PointSchema] | None = None
    grid: dict | None = None
    error_message: str | None = None
    processing_ms: int | None = None
    pipeline_timing_ms: dict[str, int] | None = None


class UploadResponse(ScanResponse):
    """Alias response for upload endpoint."""
