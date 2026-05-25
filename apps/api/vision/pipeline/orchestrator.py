"""End-to-end vision pipeline — delegates to production ScanPipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vision.board.io import decode_image_bytes
from vision.board.types import BoardDetectionResult, BoardGridResult
from vision.scanner import ScanPipeline


@dataclass(frozen=True, slots=True)
class PipelineStageTiming:
    detect_ms: int
    grid_ms: int

    @property
    def total_ms(self) -> int:
        return self.detect_ms + self.grid_ms


@dataclass(frozen=True, slots=True)
class VisionPipelineResult:
    detection: BoardDetectionResult
    grid: BoardGridResult
    timing: PipelineStageTiming
    metadata: dict[str, Any]

    @property
    def warped_board(self) -> NDArray[np.uint8]:
        return self.detection.warped_board

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    def to_metadata(self) -> dict[str, Any]:
        return {
            "detection": self.detection.to_metadata(),
            "grid": self.grid.to_metadata(),
            "timing_ms": {
                "detect": self.timing.detect_ms,
                "grid": self.timing.grid_ms,
                "total": self.timing.total_ms,
            },
            **self.metadata,
        }


class VisionPipeline:
    """Thin wrapper around ScanPipeline for upload/scan endpoints."""

    def __init__(self, scan_pipeline: ScanPipeline | None = None) -> None:
        self._scanner = scan_pipeline or ScanPipeline()

    def run(self, image: NDArray[np.uint8]) -> VisionPipelineResult:
        t0 = time.perf_counter()
        scan = self._scanner.run(image)
        total_ms = int((time.perf_counter() - t0) * 1000)

        detection = BoardDetectionResult(
            corners=scan.corners,
            homography=scan.homography,
            warped_board=scan.warped_board,
            confidence=scan.confidence,
            original_size=(scan.original_width, scan.original_height),
            output_size=scan.output_width,
        )
        grid = scan.classification.dataset_grid
        if grid is None:
            msg = "Scanner did not produce dataset grid"
            raise RuntimeError(msg)

        timing = PipelineStageTiming(
            detect_ms=scan.timing.localization_ms + scan.timing.mesh_rectify_ms,
            grid_ms=scan.timing.extraction_ms + scan.timing.occupancy_ms,
        )
        return VisionPipelineResult(
            detection=detection,
            grid=grid,
            timing=timing,
            metadata=scan.metadata,
        )

    def run_from_bytes(self, data: bytes) -> VisionPipelineResult:
        return self.run(decode_image_bytes(data))
