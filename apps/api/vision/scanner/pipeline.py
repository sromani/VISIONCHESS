"""Production-grade geometry-first chess board scanner."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vision.board.corners import order_points
from vision.board.exceptions import BoardNotFoundError
from vision.board.grid_solver import compose_board_preview
from vision.board.io import decode_image_bytes
from vision.board.split import BoardSplitResult, BoardSquareSplitter
from vision.classification.debug_viz import render_classification_grid_with_crops
from vision.classification.types import ClassificationResult
from vision.scanner.config import ScannerConfig
from vision.scanner.context import ScanContext, StageTiming
from vision.scanner.dataset.recorder import record_scan
from vision.scanner.debug.collector import build_debug_dict
from vision.scanner.stages.classification import run_classification
from vision.scanner.stages.crop_quality import build_dataset_grid, run_crop_quality
from vision.scanner.stages.extraction import run_extraction
from vision.scanner.stages.localization import run_localization
from vision.scanner.stages.mesh_rectify import run_mesh_rectification
from vision.scanner.stages.occupancy import run_occupancy
from vision.scanner.stages.validation import run_validation


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Complete scanner output — geometry is primary, warped image is derived."""

    corners: NDArray[np.float32]
    homography: NDArray[np.float64]
    warped_board: NDArray[np.uint8]
    confidence: float
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    classification: ClassificationResult
    split: BoardSplitResult | None
    debug_jpegs: dict[str, bytes]
    timing: StageTiming
    metadata: dict[str, Any]

    @property
    def corners_list(self) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.corners]

    def homography_list(self) -> list[list[float]]:
        return self.homography.astype(float).tolist()


class ScanPipeline:
    """Geometry-first: lattice → extract → enhance crops → detect → classify."""

    def __init__(self, config: ScannerConfig | None = None) -> None:
        self._config = config or ScannerConfig()

    @property
    def config(self) -> ScannerConfig:
        return self._config

    def run(
        self,
        image: NDArray[np.uint8],
        *,
        job_id: str | None = None,
        persist_dir=None,
        dataset_mode: bool | None = None,
    ) -> ScanResult:
        if image.size == 0:
            raise BoardNotFoundError("Empty image")

        config = self._config
        if dataset_mode is not None:
            config = replace(self._config, dataset_mode=dataset_mode)

        bgr = image if image.ndim == 3 else _to_bgr(image)
        orig_h, orig_w = bgr.shape[:2]
        ctx = ScanContext(original_bgr=bgr, config=config)
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

        t2b = time.perf_counter()
        run_crop_quality(ctx)
        ctx.timing.crop_quality_ms = _elapsed_ms(t2b)

        _verify_ml_models(ctx)

        t3 = time.perf_counter()
        run_occupancy(ctx)
        ctx.timing.occupancy_ms = _elapsed_ms(t3)

        t4 = time.perf_counter()
        run_classification(ctx)
        ctx.timing.classification_ms = _elapsed_ms(t4)

        from vision.scanner.stages.ml_debug import run_ml_debug

        run_ml_debug(ctx)

        build_dataset_grid(ctx)

        t5 = time.perf_counter()
        classification = run_validation(ctx)
        ctx.timing.validation_ms = _elapsed_ms(t5)

        preview = compose_board_preview(ctx.original_bgr, solve)
        split_result = None
        if persist_dir is not None:
            splitter = BoardSquareSplitter(config.grid)
            split_result = splitter.split(preview, persist_dir, board_size=config.mesh.output_size)

        if config.dataset_mode and job_id and persist_dir is not None:
            record_scan(ctx, job_id, persist_dir)

        debug_jpegs = build_debug_dict(ctx) if self._config.collect_debug else {}
        if ctx.dataset_grid is not None:
            cls_grid = render_classification_grid_with_crops(
                classification.squares,
                ctx.dataset_grid,
            )
            import cv2

            ok, buf = cv2.imencode(".jpg", cls_grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                debug_jpegs["final_board"] = buf.tobytes()

        grid_conf = ctx.grid_detection.confidence if ctx.grid_detection else 0.0
        confidence_parts = [
            grid_conf,
            max(0.0, 1.0 - solve.observed_metrics.max_error),
        ]
        if classification.confidence > 0.05:
            confidence_parts.append(classification.confidence)
        confidence = min(confidence_parts)

        extent = self._config.mesh.output_size
        ctx.metadata["pipeline"] = "geometry_first_scanner_v2"
        ctx.metadata["homography"] = solve.reference_homography.astype(float).tolist()
        ctx.record_geometry()
        ctx.metadata["timing_ms"] = {
            "localization": ctx.timing.localization_ms,
            "mesh_rectify": ctx.timing.mesh_rectify_ms,
            "extraction": ctx.timing.extraction_ms,
            "crop_quality": ctx.timing.crop_quality_ms,
            "occupancy": ctx.timing.occupancy_ms,
            "classification": ctx.timing.classification_ms,
            "validation": ctx.timing.validation_ms,
            "total": ctx.timing.total_ms,
        }

        return ScanResult(
            corners=order_points(solve.observed.outer_corners()),
            homography=solve.reference_homography,
            warped_board=preview,
            confidence=confidence,
            original_width=orig_w,
            original_height=orig_h,
            output_width=extent,
            output_height=extent,
            classification=classification,
            split=split_result,
            debug_jpegs=debug_jpegs,
            timing=ctx.timing,
            metadata=ctx.metadata,
        )

    def run_from_bytes(self, data: bytes, **kwargs) -> ScanResult:
        return self.run(decode_image_bytes(data), **kwargs)


def _to_bgr(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    import cv2

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _verify_ml_models(ctx: ScanContext) -> None:
    ml_cfg = ctx.config.ml
    if not (ml_cfg.require_occupancy_model or ml_cfg.require_piece_model):
        return
    from vision.inference.runtime import InferenceRuntime

    runtime = InferenceRuntime.load()
    if ml_cfg.require_occupancy_model and not runtime.occupancy_available:
        if not ml_cfg.allow_heuristic_fallback:
            runtime.require_all()
    if ml_cfg.require_piece_model and not runtime.piece_available:
        if not ml_cfg.allow_heuristic_fallback:
            runtime.require_all()
    ctx.metadata["ml_runtime"] = {
        "occupancy_onnx": runtime.occupancy_available,
        "piece_onnx": runtime.piece_available,
        "mode": "production" if not ml_cfg.allow_heuristic_fallback else "dev_fallback",
    }
    if ml_cfg.require_occupancy_model and runtime.occupancy_available:
        ctx.config = replace(ctx.config, occupancy=replace(ctx.config.occupancy, ml_only=True))
