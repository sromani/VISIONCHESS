"""Occupancy detection result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vision.inference.ml_debug_types import OccupancyMlDebug


@dataclass(frozen=True, slots=True)
class SquareOccupancyDebug:
    square_name: str
    is_light: bool
    foreground_score: float
    silhouette_score: float
    edge_score: float
    entropy_score: float
    center_activation: float
    ml_probability: float | None
    fused_probability: float
    occupied: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OccupancyResult:
    """Per-square occupancy decision."""

    occupied: bool
    score: float
    probability: float
    foreground_score: float
    silhouette_score: float
    edge_score: float
    entropy_score: float
    center_activation: float
    reason: str | None = None
    debug: SquareOccupancyDebug | None = None

    # Legacy fields for compatibility
    @property
    def variance(self) -> float:
        return self.center_activation

    @property
    def occupancy_ratio(self) -> float:
        return self.foreground_score

    @property
    def contour_area_ratio(self) -> float:
        return self.silhouette_score

    @property
    def edge_density(self) -> float:
        return self.edge_score


@dataclass(frozen=True, slots=True)
class OccupancyReport:
    results: dict[str, OccupancyResult]
    occupied_count: int
    empty_count: int
    board_prior_applied: bool
    board_threshold: float
    debug_rows: tuple[SquareOccupancyDebug, ...]
    ml_debug: dict[str, OccupancyMlDebug] | None = None
