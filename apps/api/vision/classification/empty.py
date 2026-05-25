"""Backward-compatible wrapper — delegates to vision.occupancy."""

from __future__ import annotations

from vision.occupancy.config import OccupancyConfig
from vision.occupancy.detector import detect_square
from vision.occupancy.types import OccupancyResult

# Legacy alias used by scanner config
EmptyDetectionConfig = OccupancyConfig


def detect_occupancy(
    crop_bgr,
    preprocessed_bgr=None,
    config: OccupancyConfig | None = None,
    *,
    row: int = 0,
    col: int = 0,
    square_name: str = "a1",
) -> OccupancyResult:
    """Detect occupancy for a single square crop."""
    source = preprocessed_bgr if preprocessed_bgr is not None else crop_bgr
    return detect_square(source, row, col, square_name, config=config)
