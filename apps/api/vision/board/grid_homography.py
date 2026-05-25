"""Grid-intersection homography — rectify from real 8×8 inner grid, not outer border."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.corners import order_points, quadrilateral_aspect_ok, quadrilateral_angles_ok
from vision.board.exceptions import BoardNotFoundError
from vision.board.grid_intersections import intersections_from_inner_corners
from vision.board.transform import (
    compute_inner_mesh_homography,
    compute_mesh_homography,
    mesh_corners_from_intersections,
)
from vision.board.types import SQUARES_PER_SIDE

INNER_CORNERS = (7, 7)
NUM_LINES = SQUARES_PER_SIDE + 1
CORNER_NAMES = ("a8", "h8", "h1", "a1")


@dataclass(frozen=True, slots=True)
class GridRectifierConfig:
    output_size: int = 800
    hough_threshold: int = 60
    min_line_length_ratio: float = 0.18
    max_line_gap_ratio: float = 0.04
    angle_tolerance_deg: float = 14.0
    min_confidence: float = 0.40
    max_aspect_deviation: float = 0.30
    min_cosine_angle: float = 0.25


@dataclass(frozen=True, slots=True)
class GridRectificationResult:
    """Homography source geometry derived from the playing grid."""

    corners: NDArray[np.float32]  # TL=a8, TR=h8, BR=h1, BL=a1
    intersections: NDArray[np.float64]  # (9, 9, 2)
    inner_corners: NDArray[np.float64] | None  # (7, 7, 2) when from chessboard
    method: str
    confidence: float
    reprojection_error: float = 0.0
    horizontal_lines: tuple[tuple[float, float, float], ...] = ()
    vertical_lines: tuple[tuple[float, float, float], ...] = ()


def detect_grid_rectification(
    gray: NDArray[np.uint8],
    config: GridRectifierConfig | None = None,
) -> GridRectificationResult:
    """Detect 8×8 playing mesh and homography source points (not photo border)."""
    cfg = config or GridRectifierConfig()
    height, width = gray.shape[:2]

    chessboard = _try_chessboard_inner_corners(gray, cfg)
    if chessboard is not None:
        return chessboard

    line_grid = _detect_hough_line_grid(gray, cfg)
    if line_grid is not None:
        intersections, inner, h_lines, v_lines, confidence, reproj = line_grid
        corners = mesh_corners_from_intersections(intersections)
        if _validate_mesh(corners, intersections, width, height, cfg):
            return GridRectificationResult(
                corners=corners,
                intersections=intersections,
                inner_corners=inner,
                method="hough_line_intersections",
                confidence=confidence,
                reprojection_error=reproj,
                horizontal_lines=tuple(_line_to_tuple(line) for line in h_lines),
                vertical_lines=tuple(_line_to_tuple(line) for line in v_lines),
            )

    msg = "Could not detect 8×8 grid intersections for homography"
    raise BoardNotFoundError(msg)


def compute_rectification_homography(
    result: GridRectificationResult,
    output_size: int,
) -> tuple[NDArray[np.float64], float]:
    """Build homography that aligns the playing mesh, not the photo frame."""
    if result.inner_corners is not None:
        return compute_inner_mesh_homography(result.inner_corners, output_size)
    return compute_mesh_homography(result.intersections, output_size)


def board_corners_from_intersections(intersections: NDArray[np.float64]) -> NDArray[np.float32]:
    """Outer grid corners in TL, TR, BR, BL order (a8, h8, h1, a1)."""
    return mesh_corners_from_intersections(intersections)


def render_grid_rectification_debug(
    image: NDArray[np.uint8],
    result: GridRectificationResult,
) -> NDArray[np.uint8]:
    """Draw detected grid lines, intersections, and a8/h8/h1/a1 on the source image."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]

    for line in result.horizontal_lines:
        _draw_line(canvas, line, (0, 255, 255), width, height, horizontal=True)
    for line in result.vertical_lines:
        _draw_line(canvas, line, (255, 255, 0), width, height, horizontal=False)

    for row in range(NUM_LINES):
        for col in range(NUM_LINES):
            x, y = result.intersections[row, col]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                cv2.circle(canvas, (xi, yi), 3, (180, 180, 180), -1, lineType=cv2.LINE_AA)

    labels = CORNER_NAMES
    colors = ((80, 80, 255), (80, 255, 80), (255, 80, 80), (80, 255, 255))
    for pt, label, color in zip(result.corners, labels, colors):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(canvas, (x, y), 10, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 10, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"rectify: {result.method} ({result.confidence:.2f})",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 255, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


@dataclass(slots=True)
class _Line2D:
    a: float
    b: float
    c: float

    def position_at(self, width: int, height: int, *, horizontal: bool) -> float:
        if horizontal:
            cx = width / 2.0
            return float(-(self.a * cx + self.c) / self.b) if abs(self.b) > 1e-6 else 0.0
        cy = height / 2.0
        return float(-(self.b * cy + self.c) / self.a) if abs(self.a) > 1e-6 else 0.0


def _try_chessboard_inner_corners(
    gray: NDArray[np.uint8],
    config: GridRectifierConfig,
) -> GridRectificationResult | None:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, INNER_CORNERS, flags)
    if not found or corners is None:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    inner = corners.reshape(INNER_CORNERS[1], INNER_CORNERS[0], 2).astype(np.float64)

    height, width = gray.shape[:2]
    intersections = intersections_from_inner_corners(inner, 0.0, 0.0)
    corners = mesh_corners_from_intersections(intersections)

    if not _validate_mesh(corners, intersections, width, height, config):
        return None

    _, reproj = compute_inner_mesh_homography(inner, config.output_size)

    return GridRectificationResult(
        corners=corners,
        intersections=intersections,
        inner_corners=inner,
        method="chessboard_inner_mesh",
        confidence=max(0.0, 1.0 - reproj / 8.0),
        reprojection_error=reproj,
    )


def _detect_hough_line_grid(
    gray: NDArray[np.uint8],
    config: GridRectifierConfig,
) -> tuple[NDArray[np.float64], None, list[_Line2D], list[_Line2D], float, float] | None:
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    min_len = int(min(width, height) * config.min_line_length_ratio)
    max_gap = int(min(width, height) * config.max_line_gap_ratio)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, config.hough_threshold),
        minLineLength=max(20, min_len),
        maxLineGap=max(4, max_gap),
    )
    if segments is None:
        return None

    horiz: list[_Line2D] = []
    vert: list[_Line2D] = []
    angle_tol = np.deg2rad(config.angle_tolerance_deg)

    for x1, y1, x2, y2 in segments[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        angle = abs(np.arctan2(dy, dx))
        line = _segment_to_line(float(x1), float(y1), float(x2), float(y2))
        if angle <= angle_tol or angle >= np.pi - angle_tol:
            horiz.append(line)
        elif abs(angle - np.pi / 2) <= angle_tol:
            vert.append(line)

    h_lines = _cluster_line_family(horiz, width, height, horizontal=True, target=NUM_LINES)
    v_lines = _cluster_line_family(vert, width, height, horizontal=False, target=NUM_LINES)
    h_lines, v_lines = _select_playing_mesh_lines(h_lines, v_lines, width, height)
    if h_lines is None or v_lines is None:
        return None

    intersections = _intersection_grid(h_lines, v_lines)
    if not _intersections_valid(intersections, width, height):
        return None

    confidence = _score_line_grid(intersections, h_lines, v_lines, width, height)
    if confidence < config.min_confidence:
        return None

    _, reproj = compute_mesh_homography(intersections, config.output_size)
    return intersections, None, h_lines, v_lines, confidence, reproj


def _segment_to_line(x1: float, y1: float, x2: float, y2: float) -> _Line2D:
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    norm = float(np.hypot(a, b))
    if norm < 1e-6:
        return _Line2D(0.0, 1.0, -y1)
    return _Line2D(a / norm, b / norm, c / norm)


def _cluster_line_family(
    lines: list[_Line2D],
    width: int,
    height: int,
    *,
    horizontal: bool,
    target: int,
) -> list[_Line2D] | None:
    if len(lines) < target - 2:
        return None

    positions = [line.position_at(width, height, horizontal=horizontal) for line in lines]
    order = np.argsort(positions)
    sorted_lines = [lines[i] for i in order]
    sorted_pos = [positions[i] for i in order]

    span = max(sorted_pos[-1] - sorted_pos[0], 1.0)
    merge_dist = max(6.0, span / (target * 2.5))

    clusters: list[list[_Line2D]] = [[sorted_lines[0]]]
    last_pos = sorted_pos[0]
    for line, pos in zip(sorted_lines[1:], sorted_pos[1:], strict=True):
        if pos - last_pos <= merge_dist:
            clusters[-1].append(line)
        else:
            clusters.append([line])
            last_pos = pos

    if len(clusters) < target - 2:
        return None

    merged = [_average_lines(group) for group in clusters]

    if len(merged) > target:
        indices = np.linspace(0, len(merged) - 1, target)
        merged = [merged[int(round(i))] for i in indices]
    elif len(merged) < target:
        lo = merged[0].position_at(width, height, horizontal=horizontal)
        hi = merged[-1].position_at(width, height, horizontal=horizontal)
        merged = []
        for i in range(target):
            t = i / (target - 1)
            pos = lo + (hi - lo) * t
            if horizontal:
                merged.append(_Line2D(0.0, 1.0, -pos))
            else:
                merged.append(_Line2D(1.0, 0.0, -pos))

    merged.sort(key=lambda line: line.position_at(width, height, horizontal=horizontal))
    return merged


def _average_lines(lines: list[_Line2D]) -> _Line2D:
    arr = np.array([[line.a, line.b, line.c] for line in lines], dtype=np.float64)
    mean = arr.mean(axis=0)
    norm = float(np.hypot(mean[0], mean[1]))
    if norm < 1e-6:
        return lines[0]
    mean /= norm
    return _Line2D(float(mean[0]), float(mean[1]), float(mean[2]))


def _intersect_lines(l1: _Line2D, l2: _Line2D) -> tuple[float, float]:
    det = l1.a * l2.b - l2.a * l1.b
    if abs(det) < 1e-8:
        return 0.0, 0.0
    x = (l1.b * l2.c - l2.b * l1.c) / det
    y = (l2.a * l1.c - l1.a * l2.c) / det
    return float(x), float(y)


def _intersection_grid(h_lines: list[_Line2D], v_lines: list[_Line2D]) -> NDArray[np.float64]:
    grid = np.zeros((NUM_LINES, NUM_LINES, 2), dtype=np.float64)
    for row, h_line in enumerate(h_lines):
        for col, v_line in enumerate(v_lines):
            grid[row, col] = _intersect_lines(h_line, v_line)
    return grid


def _intersections_valid(intersections: NDArray[np.float64], width: int, height: int) -> bool:
    if not np.all(np.isfinite(intersections)):
        return False
    margin = -0.15
    max_w = width * (1.0 - margin)
    max_h = height * (1.0 - margin)
    min_w = width * margin
    min_h = height * margin
    if np.any(intersections[:, :, 0] < min_w) or np.any(intersections[:, :, 0] > max_w):
        return False
    if np.any(intersections[:, :, 1] < min_h) or np.any(intersections[:, :, 1] > max_h):
        return False

    corners = board_corners_from_intersections(intersections)
    area = float(cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32)))
    if area < width * height * 0.04:
        return False
    return True


def _score_line_grid(
    intersections: NDArray[np.float64],
    h_lines: list[_Line2D],
    v_lines: list[_Line2D],
    width: int,
    height: int,
) -> float:
    h_pos = [line.position_at(width, height, horizontal=True) for line in h_lines]
    v_pos = [line.position_at(width, height, horizontal=False) for line in v_lines]
    h_sp = np.diff(h_pos)
    v_sp = np.diff(v_pos)
    if np.any(h_sp <= 0) or np.any(v_sp <= 0):
        return 0.0

    h_uniformity = 1.0 - min(1.0, float(np.std(h_sp) / max(np.mean(h_sp), 1.0)))
    v_uniformity = 1.0 - min(1.0, float(np.std(v_sp) / max(np.mean(v_sp), 1.0)))

    corners = board_corners_from_intersections(intersections)
    aspect = _quad_aspect_score(corners)
    return float(0.35 * h_uniformity + 0.35 * v_uniformity + 0.30 * aspect)


def _quad_aspect_score(corners: NDArray[np.float32]) -> float:
    width = float(
        max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3]))
    )
    height = float(
        max(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1]))
    )
    if width <= 0 or height <= 0:
        return 0.0
    ratio = min(width, height) / max(width, height)
    return max(0.0, min(1.0, ratio))


def _select_playing_mesh_lines(
    h_lines: list[_Line2D] | None,
    v_lines: list[_Line2D] | None,
    width: int,
    height: int,
) -> tuple[list[_Line2D] | None, list[_Line2D] | None]:
    """Pick 9×9 playing-area lines, rejecting photo/frame edges hugging the image border."""
    if h_lines is None or v_lines is None:
        return None, None
    if len(h_lines) != NUM_LINES or len(v_lines) != NUM_LINES:
        return h_lines, v_lines

    border_margin = 0.035
    h_pos = [line.position_at(width, height, horizontal=True) for line in h_lines]
    v_pos = [line.position_at(width, height, horizontal=False) for line in v_lines]

    # If outer lines sit on the image edge, shift to the next interior spacing band.
    if h_pos[0] < height * border_margin or h_pos[-1] > height * (1.0 - border_margin):
        h_lines = _shift_lines_inward(h_lines, width, height, horizontal=True)
    if v_pos[0] < width * border_margin or v_pos[-1] > width * (1.0 - border_margin):
        v_lines = _shift_lines_inward(v_lines, width, height, horizontal=False)

    return h_lines, v_lines


def _shift_lines_inward(
    lines: list[_Line2D],
    width: int,
    height: int,
    *,
    horizontal: bool,
) -> list[_Line2D]:
    """Drop one outer photo-edge line and insert one on the interior side."""
    positions = [line.position_at(width, height, horizontal=horizontal) for line in lines]
    spacings = np.diff(positions)
    if spacings.size == 0:
        return lines
    mean_spacing = float(np.mean(spacings))

    if horizontal:
        lo, hi = positions[0], positions[-1]
        if lo < height * 0.05:
            lo += mean_spacing
        if hi > height * 0.95:
            hi -= mean_spacing
        return [_Line2D(0.0, 1.0, -pos) for pos in np.linspace(lo, hi, NUM_LINES)]
    lo, hi = positions[0], positions[-1]
    if lo < width * 0.05:
        lo += mean_spacing
    if hi > width * 0.95:
        hi -= mean_spacing
    return [_Line2D(1.0, 0.0, -pos) for pos in np.linspace(lo, hi, NUM_LINES)]


def _validate_mesh(
    corners: NDArray[np.float32],
    intersections: NDArray[np.float64],
    width: int,
    height: int,
    config: GridRectifierConfig,
) -> bool:
    if not _validate_corners(corners, width, height, config):
        return False
    return _mesh_cell_uniformity_ok(intersections)


def _mesh_cell_uniformity_ok(intersections: NDArray[np.float64]) -> bool:
    """Reject photo-border quads where outer cells are much larger than inner ones."""
    widths: list[float] = []
    heights: list[float] = []
    for row in range(SQUARES_PER_SIDE):
        for col in range(SQUARES_PER_SIDE):
            tl = intersections[row, col]
            tr = intersections[row, col + 1]
            bl = intersections[row + 1, col]
            widths.append(float(np.linalg.norm(tr - tl)))
            heights.append(float(np.linalg.norm(bl - tl)))

    if not widths:
        return False

    mean_w = float(np.mean(widths))
    mean_h = float(np.mean(heights))
    if mean_w <= 0 or mean_h <= 0:
        return False

    outer = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for row, col in outer:
        tl = intersections[row, col]
        tr = intersections[row, col + 1]
        bl = intersections[row + 1, col]
        cell_w = float(np.linalg.norm(tr - tl))
        cell_h = float(np.linalg.norm(bl - tl))
        if cell_w > mean_w * 1.45 or cell_h > mean_h * 1.45:
            return False
    return True


def _validate_corners(
    corners: NDArray[np.float32],
    width: int,
    height: int,
    config: GridRectifierConfig,
) -> bool:
    if not quadrilateral_aspect_ok(corners, config.max_aspect_deviation):
        return False
    if not quadrilateral_angles_ok(corners, config.min_cosine_angle):
        return False
    area = float(cv2.contourArea(corners.reshape(-1, 1, 2)))
    if area < width * height * 0.04:
        return False
    return True


def _line_to_tuple(line: _Line2D) -> tuple[float, float, float]:
    return (line.a, line.b, line.c)


def _draw_line(
    canvas: NDArray[np.uint8],
    line: tuple[float, float, float],
    color: tuple[int, int, int],
    width: int,
    height: int,
    *,
    horizontal: bool,
) -> None:
    a, b, c = line
    pts: list[tuple[int, int]] = []
    for x in (0, width - 1):
        if abs(b) > 1e-6:
            y = -(a * x + c) / b
            if 0 <= y < height:
                pts.append((int(x), int(round(y))))
    for y in (0, height - 1):
        if abs(a) > 1e-6:
            x = -(b * y + c) / a
            if 0 <= x < width:
                pts.append((int(round(x)), int(y)))
    if len(pts) >= 2:
        cv2.line(canvas, pts[0], pts[1], color, 1, cv2.LINE_AA)
