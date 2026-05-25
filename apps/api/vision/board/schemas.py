"""Pydantic schemas for FastAPI responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from vision.board.types import BoardDetectionResult


class PointSchema(BaseModel):
    x: float
    y: float


class BoardDetectionResponse(BaseModel):
    """API-safe detection payload (no raw image bytes)."""

    corners: list[PointSchema] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Ordered TL → TR → BR → BL in original image coordinates.",
    )
    homography: list[list[float]] = Field(
        ...,
        description="3×3 perspective transform from original image to warped board.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    original_width: int = Field(..., gt=0)
    original_height: int = Field(..., gt=0)
    output_size: int = Field(..., gt=0)

    @classmethod
    def from_result(cls, result: BoardDetectionResult) -> BoardDetectionResponse:
        return cls(
            corners=[PointSchema(x=x, y=y) for x, y in result.corners_list],
            homography=result.homography_list(),
            confidence=result.confidence,
            original_width=result.original_size[0],
            original_height=result.original_size[1],
            output_size=result.output_size,
        )
