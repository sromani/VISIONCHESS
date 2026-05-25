"""Multi-signal occupancy detection with empty-square background modeling."""

from vision.occupancy.config import OccupancyConfig
from vision.occupancy.detector import OccupancyDetector, detect_board_occupancy
from vision.occupancy.types import OccupancyReport, OccupancyResult, SquareOccupancyDebug

__all__ = [
    "OccupancyConfig",
    "OccupancyDetector",
    "OccupancyReport",
    "OccupancyResult",
    "SquareOccupancyDebug",
    "detect_board_occupancy",
]
