"""Chessboard detection for real-world images.

Pipeline:
    grayscale → grid line / inner-corner detection →
    9×9 intersections → piecewise mesh rectification → uniform top-down view
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.contours import QuadCandidate, find_contours, find_quad_candidates
from vision.board.corners import order_points, scale_corners
from vision.board.detection_debug import DetectionDebugImages, build_detection_debug
from vision.board.exceptions import BoardNotFoundError
from vision.board.grid_homography import (
    GridRectificationResult,
    GridRectifierConfig,
    detect_grid_rectification,
    render_grid_rectification_debug,
)
from vision.board.io import decode_image_bytes, resize_for_detection, to_grayscale
from vision.board.mesh_rectification import (
    mesh_stats_metadata,
    rectify_board_mesh,
    render_mesh_on_image,
    render_mesh_quality_panel,
)
from vision.board.transform import scale_intersections

Point2D = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ChessboardDetectorConfig:
    output_size: int = 800
    max_detection_dim: int = 1600
    blur_kernel: int = 5
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    canny_low: int = 50
    canny_high: int = 150
    dilate_iterations: int = 1
    min_area_ratio: float = 0.08
    max_area_ratio: float = 0.92
    approx_epsilon_ratio: float = 0.02
    max_aspect_ratio_deviation: float = 0.25
    min_score: float = 0.35
    min_cosine_angle: float = 0.25
    allow_contour_fallback: bool = False


@dataclass(frozen=True, slots=True)
class ChessboardDetectionResult:
    corners: NDArray[np.float32]
    homography: NDArray[np.float64]
    warped_image: NDArray[np.uint8]
    confidence: float
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    rectification_method: str = "mesh_rectification"
    mesh_rectified: bool = True
    mesh_stats: dict[str, Any] | None = None
    debug: DetectionDebugImages | None = None

    @property
    def corners_list(self) -> list[Point2D]:
        return [(float(x), float(y)) for x, y in self.corners]

    def homography_list(self) -> list[list[float]]:
        return self.homography.astype(float).tolist()

    def to_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "corners": self.corners_list,
            "corner_labels": ["a8", "h8", "h1", "a1"],
            "homography": self.homography_list(),
            "confidence": round(self.confidence, 4),
            "rectification_method": self.rectification_method,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
        }
        if self.debug is not None:
            meta["debug_steps"] = list(self.debug.as_dict().keys())
        if self.mesh_stats is not None:
            meta["mesh_stats"] = self.mesh_stats
        meta["mesh_rectified"] = self.mesh_rectified
        return meta


class ChessboardDetector:
    """Detect a chess board via inner grid geometry and return a rectified top-down view."""

    def __init__(self, config: ChessboardDetectorConfig | None = None) -> None:
        self._config = config or ChessboardDetectorConfig()

    @property
    def config(self) -> ChessboardDetectorConfig:
        return self._config

    def detect(
        self,
        image: NDArray[np.uint8],
        *,
        collect_debug: bool = True,
    ) -> ChessboardDetectionResult:
        if image.size == 0:
            raise BoardNotFoundError("Empty image")

        original = _ensure_bgr(image)
        orig_h, orig_w = original.shape[:2]

        scaled, scale = resize_for_detection(original, self._config.max_detection_dim)
        gray = to_grayscale(scaled)
        blurred = _gaussian_blur(gray, self._config.blur_kernel)

        rectifier_cfg = GridRectifierConfig(
            output_size=self._config.output_size,
            max_aspect_deviation=self._config.max_aspect_ratio_deviation,
            min_cosine_angle=self._config.min_cosine_angle,
        )

        grid_result = detect_grid_rectification(blurred, rectifier_cfg)
        inner_corners = None
        if grid_result.inner_corners is not None:
            inner_corners = grid_result.inner_corners.copy()
            inner_corners[:, :, 0] /= scale
            inner_corners[:, :, 1] /= scale

        grid_scaled = GridRectificationResult(
            corners=scale_corners(grid_result.corners, scale),
            intersections=scale_intersections(grid_result.intersections, scale),
            inner_corners=inner_corners,
            method=grid_result.method,
            confidence=grid_result.confidence,
            reprojection_error=grid_result.reprojection_error,
            horizontal_lines=grid_result.horizontal_lines,
            vertical_lines=grid_result.vertical_lines,
        )

        mesh_result = rectify_board_mesh(
            original,
            grid_scaled.intersections,
            self._config.output_size,
        )
        corners = grid_scaled.corners
        homography = mesh_result.reference_homography
        confidence = grid_result.confidence
        method = f"mesh_{grid_result.method}"
        warped = mesh_result.warped_image
        mesh_stats = mesh_stats_metadata(mesh_result.source_stats, mesh_result.rectified_stats)
        if grid_result.reprojection_error > 0:
            confidence = min(confidence, max(0.0, 1.0 - grid_result.reprojection_error / 10.0))
        confidence = min(confidence, max(0.0, 1.0 - mesh_result.source_stats.max_error))

        debug: DetectionDebugImages | None = None
        if collect_debug:
            binary = _adaptive_threshold(blurred, self._config)
            edges = _detect_edges(binary, blurred, self._config)
            grid_debug = render_grid_rectification_debug(scaled, grid_result)
            mesh_original = render_mesh_on_image(
                original,
                grid_scaled.intersections,
                title="source mesh",
                stats=mesh_result.source_stats,
            )
            mesh_rectified = render_mesh_on_image(
                warped,
                mesh_result.rectified_intersections,
                title="rectified mesh",
                stats=mesh_result.rectified_stats,
            )
            mesh_quality = render_mesh_quality_panel(
                mesh_result.source_stats,
                mesh_result.rectified_stats,
            )
            debug = build_detection_debug(
                original=scaled,
                edges=edges,
                contours=find_contours(edges),
                candidates=[],
                selected=None,
                warped=warped,
                output_size=mesh_result.output_size,
                grid_corners=grid_result.corners,
                grid_overlay=grid_debug,
                mesh_original=mesh_original,
                mesh_rectified=mesh_rectified,
                mesh_quality=mesh_quality,
            )

        size = mesh_result.output_size
        return ChessboardDetectionResult(
            corners=order_points(corners),
            homography=homography,
            warped_image=warped,
            confidence=confidence,
            original_width=orig_w,
            original_height=orig_h,
            output_width=size,
            output_height=size,
            rectification_method=method,
            mesh_rectified=True,
            mesh_stats=mesh_stats,
            debug=debug,
        )

    def detect_from_bytes(self, data: bytes, *, collect_debug: bool = True) -> ChessboardDetectionResult:
        return self.detect(decode_image_bytes(data), collect_debug=collect_debug)


def _ensure_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _gaussian_blur(gray: NDArray[np.uint8], kernel: int) -> NDArray[np.uint8]:
    k = kernel if kernel % 2 == 1 else kernel + 1
    if k <= 1:
        return gray
    return cv2.GaussianBlur(gray, (k, k), 0)


def _adaptive_threshold(gray: NDArray[np.uint8], config: ChessboardDetectorConfig) -> NDArray[np.uint8]:
    block = config.adaptive_block_size
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        config.adaptive_c,
    )


def _detect_edges(
    binary: NDArray[np.uint8],
    blurred: NDArray[np.uint8],
    config: ChessboardDetectorConfig,
) -> NDArray[np.uint8]:
    edges = cv2.Canny(binary, config.canny_low, config.canny_high)
    if np.count_nonzero(edges) < 100:
        edges = cv2.Canny(blurred, config.canny_low, config.canny_high)
    return _dilate(edges, config.dilate_iterations)


def _dilate(edges: NDArray[np.uint8], iterations: int) -> NDArray[np.uint8]:
    if iterations <= 0:
        return edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edges, kernel, iterations=iterations)
