"""Piece-detection-only pipeline — geometry + YOLO bbox detection, no FEN."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.corners import order_points
from vision.board.exceptions import BoardNotFoundError
from vision.board.io import decode_image_bytes
from vision.scanner.config import ScannerConfig
from vision.scanner.context import ScanContext, StageTiming
from vision.scanner.debug.collector import build_debug_dict
from vision.scanner.mode import ScannerMode
from vision.scanner.stages.extraction import run_extraction
from vision.scanner.stages.localization import run_localization
from vision.scanner.stages.mesh_rectify import run_mesh_rectification
from vision.scanner.stages.yolo_detection import run_yolo_detection


@dataclass(frozen=True, slots=True)
class PieceDetectionResult:
    corners: NDArray[np.float32]
    homography: NDArray[np.float64]
    rectified_board: NDArray[np.uint8]
    confidence: float
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    debug_jpegs: dict[str, bytes]
    timing: StageTiming
    metadata: dict[str, Any]

    @property
    def corners_list(self) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.corners]

    def homography_list(self) -> list[list[float]]:
        return self.homography.astype(float).tolist()


class PieceDetectionPipeline:
    """Localize → rectify → upscale → YOLO piece detection → square assignment."""

    def __init__(self, config: ScannerConfig | None = None) -> None:
        from dataclasses import replace

        base = config or ScannerConfig.piece_detection_only()
        self._config = replace(base, mode=ScannerMode.PIECE_DETECTION_ONLY)

    @property
    def config(self) -> ScannerConfig:
        return self._config

    def run(self, image: NDArray[np.uint8]) -> PieceDetectionResult:
        if image.size == 0:
            raise BoardNotFoundError("Empty image")

        bgr = image if image.ndim == 3 else _to_bgr(image)
        orig_h, orig_w = bgr.shape[:2]
        ctx = ScanContext(original_bgr=bgr, config=self._config)
        ctx.add_debug("original", bgr)

        t0 = time.perf_counter()
        run_localization(ctx)
        ctx.timing.localization_ms = _elapsed_ms(t0)

        t1 = time.perf_counter()
        solve = run_mesh_rectification(ctx)
        ctx.timing.mesh_rectify_ms = _elapsed_ms(t1)

        t2 = time.perf_counter()
        run_extraction(ctx)
        ctx.timing.extraction_ms = _elapsed_ms(t2)

        t3 = time.perf_counter()
        run_yolo_detection(ctx)
        ctx.timing.classification_ms = _elapsed_ms(t3)

        rectified = ctx.rectified_board
        if rectified is None:
            raise BoardNotFoundError("Rectified board missing after extraction")

        debug_jpegs = build_debug_dict(ctx) if self._config.collect_debug else {}

        grid_conf = ctx.grid_detection.confidence if ctx.grid_detection else 0.0
        confidence = min(grid_conf, max(0.0, 1.0 - solve.observed_metrics.max_error))

        extent = rectified.shape[0]
        ctx.metadata["pipeline"] = "yolo_piece_detection_v1"
        ctx.metadata["mode"] = ScannerMode.PIECE_DETECTION_ONLY
        ctx.metadata["homography"] = solve.reference_homography.astype(float).tolist()
        ctx.record_geometry()
        ctx.metadata["timing_ms"] = {
            "localization": ctx.timing.localization_ms,
            "mesh_rectify": ctx.timing.mesh_rectify_ms,
            "extraction": ctx.timing.extraction_ms,
            "yolo_detection": ctx.timing.classification_ms,
            "total": ctx.timing.total_ms,
        }

        return PieceDetectionResult(
            corners=order_points(solve.observed.outer_corners()),
            homography=solve.reference_homography,
            rectified_board=rectified,
            confidence=confidence,
            original_width=orig_w,
            original_height=orig_h,
            output_width=extent,
            output_height=extent,
            debug_jpegs=debug_jpegs,
            timing=ctx.timing,
            metadata=ctx.metadata,
        )

    def run_from_bytes(self, data: bytes) -> PieceDetectionResult:
        return self.run(decode_image_bytes(data))


def _to_bgr(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
