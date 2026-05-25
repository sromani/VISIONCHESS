"""Detect actual 8×8 grid line positions on a perspective-corrected board.

The warped image rarely aligns with a naive ``height // 8`` split — borders,
frame padding, and imprecise corner detection shift the real square boundaries.
This module locates the playing area and finds 9×9 grid lines per axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.grid_intersections import (
    intersections_from_inner_corners,
    intersections_from_lines,
    lines_from_intersections,
)
from vision.board.grid_config import GridExtractorConfig
from vision.board.types import SQUARES_PER_SIDE

INNER_CORNERS = (7, 7)
NUM_LINES = SQUARES_PER_SIDE + 1


@dataclass(frozen=True, slots=True)
class PlayingAreaBounds:
    """Axis-aligned rectangle containing the 8×8 playing surface."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class GridLineSet:
    """Nine boundary positions per axis and full 9×9 intersection grid."""

    x_lines: tuple[float, ...]
    y_lines: tuple[float, ...]
    intersections: tuple[tuple[tuple[float, float], ...], ...]
    method: str
    bounds: PlayingAreaBounds

    @property
    def avg_cell_size(self) -> float:
        arr = self.intersection_array
        top = float(np.linalg.norm(arr[0, 1] - arr[0, 0]))
        left = float(np.linalg.norm(arr[1, 0] - arr[0, 0]))
        return (top + left) / 2.0

    @property
    def intersection_array(self) -> NDArray[np.float64]:
        return np.array(self.intersections, dtype=np.float64)


def detect_grid_lines(
    image: NDArray[np.uint8],
    config: GridExtractorConfig | None = None,
) -> GridLineSet:
    """Locate 9×9 grid lines on a warped board image."""
    cfg = config or GridExtractorConfig()
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    chessboard = _try_chessboard_corners(blurred, PlayingAreaBounds(0, 0, gray.shape[1], gray.shape[0]))
    if chessboard is not None:
        return chessboard

    bounds = _find_content_bounds(blurred, cfg)
    chessboard = _try_chessboard_corners(blurred, bounds)
    if chessboard is not None:
        return chessboard

    gradient = _detect_via_gradient(blurred, bounds, cfg)
    if gradient is not None:
        return gradient

    hough = _detect_via_hough(blurred, bounds, cfg)
    if hough is not None:
        return hough

    return _fallback_proportional(bounds, "proportional_bounds")


def _to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    msg = f"Expected grayscale or BGR image, got shape {image.shape}"
    raise InvalidGridError(msg)


def _find_content_bounds(gray: NDArray[np.uint8], config: GridExtractorConfig) -> PlayingAreaBounds:
    """Find the playing-area rectangle via edge-density profiles."""
    height, width = gray.shape[:2]
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    row_density = edges.sum(axis=1, dtype=np.float64) / max(width, 1)
    col_density = edges.sum(axis=0, dtype=np.float64) / max(height, 1)

    row_threshold = max(config.min_edge_density, row_density.max() * config.edge_density_ratio)
    col_threshold = max(config.min_edge_density, col_density.max() * config.edge_density_ratio)

    row_active = np.where(row_density >= row_threshold)[0]
    col_active = np.where(col_density >= col_threshold)[0]

    if row_active.size >= 2 and col_active.size >= 2:
        y0, y1 = int(row_active[0]), int(row_active[-1]) + 1
        x0, x1 = int(col_active[0]), int(col_active[-1]) + 1
    else:
        margin = int(min(height, width) * config.fallback_margin_ratio)
        x0, y0 = margin, margin
        x1, y1 = width - margin, height - margin

    x0 = max(0, min(x0, width - 2))
    y0 = max(0, min(y0, height - 2))
    x1 = max(x0 + 2, min(x1, width))
    y1 = max(y0 + 2, min(y1, height))

    return _snap_bounds_to_edges(PlayingAreaBounds(x0=x0, y0=y0, x1=x1, y1=y1), width, height, config)


def _snap_bounds_to_edges(
    bounds: PlayingAreaBounds,
    width: int,
    height: int,
    config: GridExtractorConfig,
) -> PlayingAreaBounds:
    """Snap near-full-bleed detections to image edges to avoid losing outer rank/file."""
    inset_x = config.edge_snap_ratio
    inset_y = config.edge_snap_ratio
    x0, y0, x1, y1 = bounds.x0, bounds.y0, bounds.x1, bounds.y1

    if x0 / max(width, 1) <= inset_x:
        x0 = 0
    if y0 / max(height, 1) <= inset_y:
        y0 = 0
    if (width - x1) / max(width, 1) <= inset_x:
        x1 = width
    if (height - y1) / max(height, 1) <= inset_y:
        y1 = height

    return PlayingAreaBounds(x0=x0, y0=y0, x1=x1, y1=y1)


def _try_chessboard_corners(gray: NDArray[np.uint8], bounds: PlayingAreaBounds) -> GridLineSet | None:
    """Use OpenCV inner-corner detection when the empty pattern is visible."""
    roi = gray[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
    if roi.shape[0] < 80 or roi.shape[1] < 80:
        return None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    grid = _find_inner_corner_grid(roi, flags)
    if grid is None:
        pad = 12
        padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        grid = _find_inner_corner_grid(padded, flags, offset=(-pad, -pad))
    if grid is None:
        return None

    intersections = intersections_from_inner_corners(grid, float(bounds.x0), float(bounds.y0))
    x_lines, y_lines = lines_from_intersections(intersections)

    if not _validate_lines(x_lines, y_lines, gray.shape[1], gray.shape[0]):
        return None

    return _make_line_set(x_lines, y_lines, intersections, "chessboard_corners", bounds)


def _find_inner_corner_grid(
    image: NDArray[np.uint8],
    flags: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64] | None:
    found, corners = cv2.findChessboardCorners(image, INNER_CORNERS, flags)
    if not found or corners is None:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(image, corners, (7, 7), (-1, -1), criteria)
    grid = corners.reshape(INNER_CORNERS[1], INNER_CORNERS[0], 2).astype(np.float64)
    grid[:, :, 0] += offset[0]
    grid[:, :, 1] += offset[1]
    return grid


def _detect_via_gradient(
    gray: NDArray[np.uint8],
    bounds: PlayingAreaBounds,
    config: GridExtractorConfig,
) -> GridLineSet | None:
    """Find grid lines from Sobel edge-strength profiles inside the playing area."""
    roi = gray[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
    if roi.shape[0] < 40 or roi.shape[1] < 40:
        return None

    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

    col_profile = np.sum(np.abs(sobel_x), axis=0)
    row_profile = np.sum(np.abs(sobel_y), axis=1)

    col_profile = _smooth_profile(col_profile, config.profile_smooth_sigma)
    row_profile = _smooth_profile(row_profile, config.profile_smooth_sigma)

    x_peaks = _find_grid_peaks(col_profile, expected_inner=7)
    y_peaks = _find_grid_peaks(row_profile, expected_inner=7)

    if x_peaks is None or y_peaks is None:
        return None

    x_lines = _peaks_to_lines(x_peaks, bounds.width, bounds.x0)
    y_lines = _peaks_to_lines(y_peaks, bounds.height, bounds.y0)

    x_lines = _refine_equal_spacing(x_lines)
    y_lines = _refine_equal_spacing(y_lines)

    if not _validate_lines(x_lines, y_lines, gray.shape[1], gray.shape[0]):
        return None

    intersections = intersections_from_lines(x_lines, y_lines)
    return _make_line_set(x_lines, y_lines, intersections, "gradient_profile", bounds)


def _detect_via_hough(
    gray: NDArray[np.uint8],
    bounds: PlayingAreaBounds,
    config: GridExtractorConfig,
) -> GridLineSet | None:
    """Cluster Hough line segments into 9 horizontal and 9 vertical positions."""
    roi = gray[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
    edges = cv2.Canny(roi, 50, 150)
    min_len = int(min(roi.shape[0], roi.shape[1]) * 0.25)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, min_len // 2),
        minLineLength=min_len,
        maxLineGap=int(min(roi.shape[0], roi.shape[1]) * 0.05),
    )
    if segments is None:
        return None

    horiz: list[float] = []
    vert: list[float] = []
    angle_tol = np.deg2rad(12)

    for seg in segments[:, 0]:
        x1, y1, x2, y2 = seg
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        angle = abs(np.arctan2(dy, dx))
        if angle <= angle_tol or angle >= np.pi - angle_tol:
            horiz.append((y1 + y2) / 2.0)
        elif abs(angle - np.pi / 2) <= angle_tol:
            vert.append((x1 + x2) / 2.0)

    x_clustered = _cluster_positions(vert, NUM_LINES)
    y_clustered = _cluster_positions(horiz, NUM_LINES)

    if x_clustered is None or y_clustered is None:
        return None

    x_lines = tuple(float(v + bounds.x0) for v in x_clustered)
    y_lines = tuple(float(v + bounds.y0) for v in y_clustered)

    x_lines = _refine_equal_spacing(x_lines)
    y_lines = _refine_equal_spacing(y_lines)

    if not _validate_lines(x_lines, y_lines, gray.shape[1], gray.shape[0]):
        return None

    intersections = intersections_from_lines(x_lines, y_lines)
    return _make_line_set(x_lines, y_lines, intersections, "hough_lines", bounds)


def _fallback_proportional(bounds: PlayingAreaBounds, method: str) -> GridLineSet:
    """Divide the detected playing area into 8 equal cells (not full image // 8)."""
    x_lines = tuple(
        bounds.x0 + bounds.width * i / SQUARES_PER_SIDE for i in range(NUM_LINES)
    )
    y_lines = tuple(
        bounds.y0 + bounds.height * i / SQUARES_PER_SIDE for i in range(NUM_LINES)
    )
    intersections = intersections_from_lines(x_lines, y_lines)
    return _make_line_set(x_lines, y_lines, intersections, method, bounds)


def _make_line_set(
    x_lines: tuple[float, ...],
    y_lines: tuple[float, ...],
    intersections: NDArray[np.float64],
    method: str,
    bounds: PlayingAreaBounds,
) -> GridLineSet:
    grid = tuple(
        tuple((float(pt[0]), float(pt[1])) for pt in row)
        for row in intersections
    )
    return GridLineSet(
        x_lines=x_lines,
        y_lines=y_lines,
        intersections=grid,
        method=method,
        bounds=bounds,
    )


def _smooth_profile(profile: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    if profile.size == 0:
        return profile
    k = max(3, int(sigma * 6) | 1)
    return cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), sigma).ravel()


def _find_grid_peaks(profile: NDArray[np.float64], expected_inner: int) -> list[float] | None:
    """Return 7 inner line positions; outer lines are extrapolated later."""
    if profile.size < expected_inner + 2:
        return None

    norm = profile / max(float(profile.max()), 1.0)
    min_distance = max(3, profile.size // (expected_inner + 2))

    peaks: list[int] = []
    for idx in range(1, profile.size - 1):
        if norm[idx] < 0.15:
            continue
        if norm[idx] >= norm[idx - 1] and norm[idx] >= norm[idx + 1]:
            if peaks and idx - peaks[-1] < min_distance:
                if norm[idx] > norm[peaks[-1]]:
                    peaks[-1] = idx
            else:
                peaks.append(idx)

    if len(peaks) < expected_inner - 1:
        return None

    if len(peaks) > expected_inner:
        peaks = _select_evenly_spaced_peaks(peaks, expected_inner)

    if len(peaks) < expected_inner - 1:
        return None

    return [float(p) for p in peaks]


def _select_evenly_spaced_peaks(peaks: list[int], count: int) -> list[int]:
    """Pick ``count`` peaks that best match equal spacing."""
    if len(peaks) <= count:
        return peaks

    best: list[int] | None = None
    best_score = float("inf")

    for start in range(len(peaks) - count + 1):
        window = peaks[start : start + count]
        spacings = np.diff(window)
        score = float(np.std(spacings))
        if score < best_score:
            best_score = score
            best = window

    return best or peaks[:count]


def _peaks_to_lines(peaks: list[float], span: int, offset: int) -> tuple[float, ...]:
    """Convert inner peaks to 9 boundary lines including extrapolated edges."""
    if not peaks:
        return tuple(offset + span * i / SQUARES_PER_SIDE for i in range(NUM_LINES))

    inner = np.array(peaks, dtype=np.float64)
    if inner.size >= 2:
        spacing = float(np.mean(np.diff(inner)))
    else:
        spacing = span / SQUARES_PER_SIDE

    lines = [inner[0] - spacing + offset]
    for pos in inner:
        lines.append(pos + offset)
    lines.append(inner[-1] + spacing + offset)

    while len(lines) < NUM_LINES:
        spacing = (lines[-1] - lines[0]) / max(len(lines) - 1, 1)
        lines.append(lines[-1] + spacing)

    if len(lines) > NUM_LINES:
        indices = np.linspace(0, len(lines) - 1, NUM_LINES)
        lines = [lines[int(round(i))] for i in indices]

    return tuple(lines[:NUM_LINES])


def _cluster_positions(positions: list[float], target: int) -> list[float] | None:
    if len(positions) < target - 2:
        return None

    sorted_pos = sorted(positions)
    clusters: list[list[float]] = [[sorted_pos[0]]]

    span = sorted_pos[-1] - sorted_pos[0]
    merge_dist = max(4.0, span / (target * 2))

    for pos in sorted_pos[1:]:
        if pos - clusters[-1][-1] <= merge_dist:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])

    if len(clusters) < target - 2:
        return None

    centers = [float(np.mean(c)) for c in clusters]

    if len(centers) > target:
        indices = np.linspace(0, len(centers) - 1, target)
        centers = [centers[int(round(i))] for i in indices]
    elif len(centers) < target:
        lo, hi = centers[0], centers[-1]
        centers = [lo + (hi - lo) * i / (target - 1) for i in range(target)]

    return centers


def _refine_equal_spacing(lines: tuple[float, ...]) -> tuple[float, ...]:
    """Snap nearly-uniform lines to equal spacing while preserving center and span."""
    arr = np.array(lines, dtype=np.float64)
    if arr.size != NUM_LINES:
        return lines

    lo, hi = arr[0], arr[-1]
    ideal = np.linspace(lo, hi, NUM_LINES)
    spacings = np.diff(arr)
    if spacings.size == 0:
        return lines

    mean_spacing = float(np.mean(spacings))
    if mean_spacing <= 0:
        return lines

    deviation = float(np.std(spacings) / mean_spacing)
    if deviation > 0.12:
        return lines

    blend = 0.65
    refined = blend * ideal + (1.0 - blend) * arr
    return tuple(float(v) for v in refined)


def _validate_lines(
    x_lines: tuple[float, ...],
    y_lines: tuple[float, ...],
    width: int,
    height: int,
) -> bool:
    if len(x_lines) != NUM_LINES or len(y_lines) != NUM_LINES:
        return False

    for lines, limit in ((x_lines, width), (y_lines, height)):
        if any(not np.isfinite(v) for v in lines):
            return False
        if lines[0] < -1 or lines[-1] > limit + 1:
            return False
        diffs = np.diff(lines)
        if np.any(diffs <= 2):
            return False
        if not np.all(diffs > 0):
            return False

    return True
