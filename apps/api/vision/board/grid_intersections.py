"""9×9 grid intersection geometry for perspective-aware square extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vision.board.types import SQUARES_PER_SIDE

NUM_LINES = SQUARES_PER_SIDE + 1


def intersections_from_lines(
    x_lines: tuple[float, ...],
    y_lines: tuple[float, ...],
) -> NDArray[np.float64]:
    """Build axis-aligned intersection grid from 1-D line positions."""
    points = np.zeros((NUM_LINES, NUM_LINES, 2), dtype=np.float64)
    for row in range(NUM_LINES):
        for col in range(NUM_LINES):
            points[row, col] = (x_lines[col], y_lines[row])
    return points


def intersections_from_inner_corners(
    inner: NDArray[np.float64],
    offset_x: float,
    offset_y: float,
) -> NDArray[np.float64]:
    """Extrapolate a full 9×9 intersection grid from 7×7 inner chessboard corners."""
    full = np.zeros((NUM_LINES, NUM_LINES, 2), dtype=np.float64)

    for row in range(7):
        for col in range(7):
            full[row + 1, col + 1] = inner[row, col]

    for col in range(1, 8):
        full[0, col] = 2.0 * full[1, col] - full[2, col]
        full[8, col] = 2.0 * full[7, col] - full[6, col]

    for row in range(1, 8):
        full[row, 0] = 2.0 * full[row, 1] - full[row, 2]
        full[row, 8] = 2.0 * full[row, 7] - full[row, 6]

    full[0, 0] = full[0, 1] + full[1, 0] - full[1, 1]
    full[0, 8] = full[0, 7] + full[1, 8] - full[1, 7]
    full[8, 0] = full[8, 1] + full[7, 0] - full[7, 1]
    full[8, 8] = full[8, 7] + full[7, 8] - full[7, 7]

    full[:, :, 0] += offset_x
    full[:, :, 1] += offset_y
    return full


def lines_from_intersections(
    intersections: NDArray[np.float64],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Derive representative x/y line positions from a skewed intersection grid."""
    x_lines = tuple(float(np.mean(intersections[:, col, 0])) for col in range(NUM_LINES))
    y_lines = tuple(float(np.mean(intersections[row, :, 1])) for row in range(NUM_LINES))
    return x_lines, y_lines


def cell_quad(intersections: NDArray[np.float64], row: int, col: int) -> NDArray[np.float64]:
    """Return TL, TR, BR, BL corners for one board square."""
    return np.array(
        [
            intersections[row, col],
            intersections[row, col + 1],
            intersections[row + 1, col + 1],
            intersections[row + 1, col],
        ],
        dtype=np.float64,
    )


def inset_quad(quad: NDArray[np.float64], margin_ratio: float) -> NDArray[np.float64]:
    """Shrink a quadrilateral toward its centroid by ``margin_ratio`` per side."""
    center = quad.mean(axis=0)
    scale = max(0.05, 1.0 - 2.0 * margin_ratio)
    return center + scale * (quad - center)


def quad_envelope(quad: NDArray[np.float64]) -> tuple[int, int, int, int]:
    """Axis-aligned envelope of a quad — metadata only, never used for cropping."""
    xs = quad[:, 0]
    ys = quad[:, 1]
    return (
        int(np.floor(xs.min())),
        int(np.floor(ys.min())),
        int(np.ceil(xs.max())),
        int(np.ceil(ys.max())),
    )


def quad_to_tuple(quad: NDArray[np.float64]) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in quad)
