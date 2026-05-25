"""Debug visualizations for board grid splitting."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.types import BoardGridResult, SQUARES_PER_SIDE, SquareCrop


def render_grid_overlay(
    warped_board: NDArray[np.uint8],
    grid: BoardGridResult,
) -> NDArray[np.uint8]:
    """Draw detected grid lines, quad boundaries, and algebraic labels."""
    canvas = warped_board.copy()

    if grid.x_lines and grid.y_lines:
        for y in grid.y_lines:
            yi = int(round(y))
            cv2.line(canvas, (0, yi), (canvas.shape[1] - 1, yi), (0, 255, 255), 1)
        for x in grid.x_lines:
            xi = int(round(x))
            cv2.line(canvas, (xi, 0), (xi, canvas.shape[0] - 1), (0, 255, 255), 1)

    for sq in grid.flat:
        cell_poly = _quad_to_int32(sq.cell_quad)
        cv2.polylines(canvas, [cell_poly], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        crop_poly = _quad_to_int32(sq.crop_quad)
        cv2.polylines(canvas, [crop_poly], isClosed=True, color=(0, 180, 255), thickness=1, lineType=cv2.LINE_AA)

        center = _quad_center(sq.cell_quad)
        _draw_centered_label(canvas, sq.square_name, center, scale=0.4, color=(255, 80, 80))

    if grid.grid_method:
        cv2.putText(
            canvas,
            f"grid: {grid.grid_method}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )

    return canvas


def render_grid_debug_extreme(
    warped_board: NDArray[np.uint8],
    grid: BoardGridResult,
) -> NDArray[np.uint8]:
    """Extreme debug overlay: lines, square quads, centers, and a1-h8 labels."""
    base = warped_board.copy()
    height, width = base.shape[:2]
    tint = np.zeros_like(base)

    for sq in grid.flat:
        polygon = _quad_to_int32(sq.cell_quad)
        fill = (40, 180, 40) if (sq.row + sq.col) % 2 == 0 else (180, 60, 40)
        cv2.fillPoly(tint, [polygon], fill)

    canvas = cv2.addWeighted(base, 0.55, tint, 0.45, 0)

    for sq in grid.flat:
        cell_poly = _quad_to_int32(sq.cell_quad)
        cv2.polylines(canvas, [cell_poly], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [cell_poly], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        crop_poly = _quad_to_int32(sq.crop_quad)
        cv2.polylines(canvas, [crop_poly], isClosed=True, color=(0, 220, 255), thickness=1, lineType=cv2.LINE_AA)

        center = _quad_center(sq.cell_quad)
        cv2.drawMarker(
            canvas,
            center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(canvas, center, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        _draw_centered_label(canvas, sq.square_name, center, scale=0.55, color=(255, 255, 255))

    if grid.x_lines and grid.y_lines:
        for idx, x in enumerate(grid.x_lines):
            xi = int(round(x))
            cv2.line(canvas, (xi, 0), (xi, height - 1), (255, 255, 0), 2, cv2.LINE_AA)
            label_y = 16 if idx % 2 == 0 else height - 8
            cv2.putText(
                canvas,
                f"x{idx}={x:.1f}",
                (min(xi + 2, width - 72), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        for idx, y in enumerate(grid.y_lines):
            yi = int(round(y))
            cv2.line(canvas, (0, yi), (width - 1, yi), (0, 255, 255), 2, cv2.LINE_AA)
            label_x = 4 if idx % 2 == 0 else width - 72
            cv2.putText(
                canvas,
                f"y{idx}={y:.1f}",
                (label_x, min(yi + 14, height - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    _draw_legend(canvas, grid)
    return canvas


def render_crop_montage(
    grid: BoardGridResult,
    cell_px: int = 80,
    gutter: int = 2,
) -> NDArray[np.uint8]:
    """Build an 8×8 montage of all square crops with file/rank labels."""
    label_h = 16
    tile = cell_px + label_h
    grid_px = SQUARES_PER_SIDE * tile + (SQUARES_PER_SIDE + 1) * gutter
    montage = np.full((grid_px, grid_px, 3), 40, dtype=np.uint8)

    for sq in grid.flat:
        resized = cv2.resize(sq.image, (cell_px, cell_px), interpolation=cv2.INTER_AREA)
        ox = gutter + sq.col * (tile + gutter)
        oy = gutter + sq.row * (tile + gutter)
        montage[oy + label_h : oy + label_h + cell_px, ox : ox + cell_px] = resized
        cv2.putText(
            montage,
            sq.square_name,
            (ox + 4, oy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return montage


def square_polygons_metadata(grid: BoardGridResult) -> list[dict[str, object]]:
    """Exact square quads for API metadata / programmatic checks."""
    polygons: list[dict[str, object]] = []
    for sq in grid.flat:
        center = _quad_center(sq.cell_quad)
        polygons.append(
            {
                "name": sq.square_name,
                "row": sq.row,
                "col": sq.col,
                "polygon": [list(pt) for pt in sq.cell_quad],
                "crop_polygon": [list(pt) for pt in sq.crop_quad],
                "center": [float(center[0]), float(center[1])],
                "cell_bbox": list(sq.cell_bbox),
                "crop_bbox": list(sq.bbox),
            }
        )
    return polygons


def _quad_to_int32(quad: tuple[tuple[float, float], ...]) -> NDArray[np.int32]:
    return np.array([[int(round(x)), int(round(y))] for x, y in quad], dtype=np.int32)


def _quad_center(quad: tuple[tuple[float, float], ...]) -> tuple[int, int]:
    arr = np.array(quad, dtype=np.float64)
    center = arr.mean(axis=0)
    return int(round(center[0])), int(round(center[1]))


def _draw_centered_label(
    canvas: NDArray[np.uint8],
    text: str,
    center: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), _baseline = cv2.getTextSize(text, font, scale, thickness)
    x = center[0] - tw // 2
    y = center[1] + th // 2
    cv2.putText(canvas, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_legend(canvas: NDArray[np.uint8], grid: BoardGridResult) -> None:
    lines = [
        "EXTREME GRID DEBUG",
        f"method: {grid.grid_method or 'unknown'}",
        "white quad = exact square",
        "cyan quad = perspective crop",
        "cross = square center",
        "yellow = x-lines | cyan = y-lines",
    ]
    y = 22
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 255, 220), 1, cv2.LINE_AA)
        y += 18
