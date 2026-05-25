"""Phase 1 — infer observed playing lattice from image features."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.corners import scale_corners
from vision.board.exceptions import BoardNotFoundError
from vision.board.grid_homography import GridRectificationResult, detect_grid_rectification
from vision.board.io import resize_for_detection, to_grayscale
from vision.board.playing_grid import GridConstraints, PlayingGrid
from vision.scanner.config import LocalizationConfig
from vision.scanner.context import ScanContext


def run_localization(ctx: ScanContext) -> PlayingGrid:
    """Detect 9×9 lattice in image space — geometry, not board-as-image."""
    cfg = ctx.config.localization

    scaled, scale = resize_for_detection(ctx.original_bgr, cfg.max_detection_dim)
    ctx.scale = scale
    gray = to_grayscale(scaled)
    gray = _clahe(gray, cfg.clahe_clip, cfg.clahe_grid)
    gray = _gaussian_blur(gray, cfg.blur_kernel)
    ctx.working_gray = gray
    ctx.edges = _detect_edges(gray, cfg)

    rectifier = ctx.config.grid_rectifier()
    limits = GridConstraints(
        max_column_width_cv=cfg.max_source_spacing_cv,
        max_row_height_cv=cfg.max_source_spacing_cv,
        max_orthogonality_deg=cfg.max_source_orthogonality_deg,
    )
    last_error: Exception | None = None

    for attempt in range(cfg.max_mesh_retries + 1):
        try:
            grid = detect_grid_rectification(gray, rectifier)
            intersections = _scale_intersections(grid.intersections, scale)
            observed = PlayingGrid.from_intersections(intersections)
            metrics = observed.metrics()

            if not metrics.satisfies(limits) and attempt < cfg.max_mesh_retries:
                rectifier = _relax_rectifier(rectifier, attempt)
                continue

            ctx.grid_detection = grid
            ctx.observed_grid = observed
            ctx.metadata["localization"] = {
                "method": grid.method,
                "confidence": round(grid.confidence, 4),
                "attempt": attempt,
                "constraints": metrics.to_dict(),
            }

            if ctx.config.collect_debug:
                ctx.add_debug("detected_lines", _render_lattice(ctx.original_bgr, observed, "observed lattice"))
                ctx.add_debug("intersections", _render_points(ctx.original_bgr, observed))
            return observed
        except BoardNotFoundError as exc:
            last_error = exc
            if attempt < cfg.max_mesh_retries:
                rectifier = _relax_rectifier(rectifier, attempt)
                continue
            raise

    if last_error is not None:
        raise last_error
    msg = "Could not localize playing lattice"
    raise BoardNotFoundError(msg)


def _scale_intersections(intersections: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    if scale == 1.0:
        return intersections.copy()
    scaled = intersections.copy()
    scaled[:, :, 0] /= scale
    scaled[:, :, 1] /= scale
    return scaled


def _relax_rectifier(rectifier, attempt: int):
    from vision.board.grid_homography import GridRectifierConfig

    return GridRectifierConfig(
        output_size=rectifier.output_size,
        hough_threshold=max(25, rectifier.hough_threshold - 10 * (attempt + 1)),
        min_confidence=max(0.25, rectifier.min_confidence - 0.05 * (attempt + 1)),
        max_aspect_deviation=min(0.45, rectifier.max_aspect_deviation + 0.05),
        min_cosine_angle=max(0.15, rectifier.min_cosine_angle - 0.05),
        angle_tolerance_deg=rectifier.angle_tolerance_deg + 2.0,
        min_line_length_ratio=rectifier.min_line_length_ratio,
        max_line_gap_ratio=rectifier.max_line_gap_ratio,
    )


def _clahe(gray: NDArray[np.uint8], clip: float, grid: int) -> NDArray[np.uint8]:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def _gaussian_blur(gray: NDArray[np.uint8], kernel: int) -> NDArray[np.uint8]:
    k = kernel if kernel % 2 == 1 else kernel + 1
    if k <= 1:
        return gray
    return cv2.GaussianBlur(gray, (k, k), 0)


def _detect_edges(gray: NDArray[np.uint8], cfg: LocalizationConfig) -> NDArray[np.uint8]:
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edges, kernel, iterations=1)


def _render_lattice(image: NDArray[np.uint8], grid: PlayingGrid, title: str) -> NDArray[np.uint8]:
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    pts = grid.intersections

    for row in range(9):
        p0, p1 = pts[row, 0], pts[row, 8]
        cv2.line(canvas, _pt(p0), _pt(p1), (0, 255, 255), 1, cv2.LINE_AA)
    for col in range(9):
        p0, p1 = pts[0, col], pts[8, col]
        cv2.line(canvas, _pt(p0), _pt(p1), (255, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1, cv2.LINE_AA)
    m = grid.metrics()
    cv2.putText(
        canvas,
        f"orth={m.orthogonality_error_deg:.1f}° col_cv={m.column_width_cv:.3f}",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _render_points(image: NDArray[np.uint8], grid: PlayingGrid) -> NDArray[np.uint8]:
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    for row in range(9):
        for col in range(9):
            x, y = grid.point(row, col)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(canvas, (xi, yi), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return canvas


def _pt(p: NDArray[np.float64]) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))
