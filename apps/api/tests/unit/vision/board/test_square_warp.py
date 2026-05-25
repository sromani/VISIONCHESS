"""Tests for perspective square warping."""

from __future__ import annotations

import cv2
import numpy as np

from vision.board.grid import BoardGridExtractor
from vision.board.grid_intersections import cell_quad, inset_quad, intersections_from_lines
from vision.board.square_warp import quad_output_size, warp_quad_to_square


def _checkerboard(size: int = 800) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size / 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 80
            y0 = int(round(row * cell))
            y1 = int(round((row + 1) * cell))
            x0 = int(round(col * cell))
            x1 = int(round((col + 1) * cell))
            board[y0:y1, x0:x1] = (color, color, color)
    return board


def _residual_perspective_board(size: int = 800) -> np.ndarray:
    """Board with residual trapezoid perspective after a partial warp."""
    board = _checkerboard(size)
    src = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    dst = np.float32([[35, 18], [765, 28], [725, 782], [28, 772]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board, matrix, (size, size), borderValue=(15, 15, 15))


class TestSquareWarp:
    def test_warp_quad_produces_square_image(self) -> None:
        image = _checkerboard(800)
        x_lines = tuple(i * 100.0 for i in range(9))
        y_lines = tuple(i * 100.0 for i in range(9))
        intersections = intersections_from_lines(x_lines, y_lines)
        quad = cell_quad(intersections, 0, 0)
        crop = warp_quad_to_square(image, inset_quad(quad, 0.08), quad_output_size(quad))
        assert crop.shape[0] == crop.shape[1]
        assert crop.shape[0] >= 80

    def test_perspective_board_crops_stay_uniform(self) -> None:
        grid = BoardGridExtractor().extract(_residual_perspective_board())
        for sq in grid.flat:
            assert sq.image.shape[0] == sq.image.shape[1]
            gray = cv2.cvtColor(sq.image, cv2.COLOR_BGR2GRAY)
            assert float(np.std(gray)) < 25.0

    def test_cell_quad_is_not_axis_aligned_on_skew(self) -> None:
        grid = BoardGridExtractor().extract(_residual_perspective_board())
        e4 = grid.crop_at(4, 4)
        quad = np.array(e4.cell_quad, dtype=np.float64)
        top = quad[1] - quad[0]
        left = quad[3] - quad[0]
        assert abs(top[1]) > 0.5 or abs(left[0]) > 0.5
