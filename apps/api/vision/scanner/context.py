"""Mutable scan context — geometry first, image second."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vision.board.grid_homography import GridRectificationResult
from vision.board.grid_solver import GridSolveResult
from vision.board.playing_grid import PlayingGrid
from vision.board.types import BoardGridResult
from vision.occupancy.types import OccupancyResult
from vision.classification.types import OrientationCandidate, SquareClassification
from vision.inference.ml_debug_types import MlDebugReport, OccupancyMlDebug, PieceMlDebug
from vision.scanner.config import ScannerConfig


@dataclass
class StageTiming:
    localization_ms: int = 0
    mesh_rectify_ms: int = 0
    extraction_ms: int = 0
    crop_quality_ms: int = 0
    occupancy_ms: int = 0
    classification_ms: int = 0
    validation_ms: int = 0

    @property
    def total_ms(self) -> int:
        return (
            self.localization_ms
            + self.mesh_rectify_ms
            + self.extraction_ms
            + self.crop_quality_ms
            + self.occupancy_ms
            + self.classification_ms
            + self.validation_ms
        )


@dataclass
class ScanContext:
    """Pipeline state: the playing grid is primary; the image is a sample source."""

    original_bgr: NDArray[np.uint8]
    config: ScannerConfig
    scale: float = 1.0
    working_gray: NDArray[np.uint8] | None = None
    edges: NDArray[np.uint8] | None = None
    grid_detection: GridRectificationResult | None = None
    observed_grid: PlayingGrid | None = None
    grid_solve: GridSolveResult | None = None
    rectified_board: NDArray[np.uint8] | None = None
    raw_grid: BoardGridResult | None = None
    analysis_grid: BoardGridResult | None = None
    context_grid: BoardGridResult | None = None
    dataset_grid: BoardGridResult | None = None
    occupancy: dict[str, OccupancyResult] = field(default_factory=dict)
    squares: list[SquareClassification] = field(default_factory=list)
    candidates: list[OrientationCandidate] = field(default_factory=list)
    timing: StageTiming = field(default_factory=StageTiming)
    debug_frames: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    ml_debug_report: MlDebugReport | None = None
    piece_ml_debug_map: dict[str, PieceMlDebug] = field(default_factory=dict)
    context_piece_ml_debug_map: dict[str, PieceMlDebug] = field(default_factory=dict)
    occupancy_ml_debug_map: dict[str, OccupancyMlDebug] = field(default_factory=dict)

    @property
    def canonical_grid(self) -> PlayingGrid | None:
        return None if self.grid_solve is None else self.grid_solve.canonical

    @property
    def warped_board(self) -> NDArray[np.uint8] | None:
        return self.debug_frames.get("rectified_board")

    def add_debug(self, name: str, frame: NDArray[np.uint8]) -> None:
        self.debug_frames[name] = frame

    def record_geometry(self) -> None:
        if self.observed_grid is not None:
            self.metadata["observed_grid"] = self.observed_grid.to_metadata()
        if self.grid_solve is not None:
            self.metadata["canonical_grid"] = self.grid_solve.canonical.to_metadata()
            self.metadata["grid_solve"] = {
                "constraints_satisfied": self.grid_solve.constraints_satisfied,
                "observed": self.grid_solve.observed_metrics.to_dict(),
                "canonical": self.grid_solve.canonical_metrics.to_dict(),
            }
