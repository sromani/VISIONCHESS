"""Tests for grid-intersection homography."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.board.exceptions import BoardNotFoundError
from vision.board.grid_homography import (
    board_corners_from_intersections,
    detect_grid_rectification,
)
from vision.board.grid_lines import detect_grid_lines


def _checkerboard(size: int = 800) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size / 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 70
            y0, y1 = int(row * cell), int((row + 1) * cell)
            x0, x1 = int(col * cell), int((col + 1) * cell)
            board[y0:y1, x0:x1] = (color, color, color)
    return board


def _skewed_scene(size: int = 600) -> np.ndarray:
    board = _checkerboard(size)
    src = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    dst = np.float32([[75, 40], [525, 20], [485, 560], [35, 525]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board, matrix, (600, 600), borderValue=(25, 25, 25))


class TestGridHomography:
    def test_detects_grid_on_skewed_checkerboard(self) -> None:
        gray = cv2.cvtColor(_skewed_scene(), cv2.COLOR_BGR2GRAY)
        result = detect_grid_rectification(gray)
        assert result.method in {"chessboard_inner_mesh", "hough_line_intersections"}
        assert result.confidence >= 0.4
        assert result.corners.shape == (4, 2)
        assert result.intersections.shape == (9, 9, 2)

    def test_corners_are_labeled_positions(self) -> None:
        gray = cv2.cvtColor(_skewed_scene(), cv2.COLOR_BGR2GRAY)
        result = detect_grid_rectification(gray)
        tl, tr, br, bl = result.corners
        assert tl[0] < tr[0]
        assert bl[0] < br[0]
        assert tl[1] < bl[1]
        assert tr[1] < br[1]

    def test_warp_produces_parallel_grid(self) -> None:
        from vision.board.mesh_rectification import rectify_board_mesh

        scene = _skewed_scene()
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        grid = detect_grid_rectification(gray)
        warped = rectify_board_mesh(scene, grid.intersections, 800).warped_image
        lines = detect_grid_lines(warped)
        x_sp = np.diff(lines.x_lines)
        y_sp = np.diff(lines.y_lines)
        assert float(np.std(x_sp) / np.mean(x_sp)) < 0.08
        assert float(np.std(y_sp) / np.mean(y_sp)) < 0.08

    def test_rejects_blank_image(self) -> None:
        blank = np.zeros((400, 400), dtype=np.uint8)
        with pytest.raises(BoardNotFoundError):
            detect_grid_rectification(blank)

    def test_board_corners_from_intersections_order(self) -> None:
        pts = np.zeros((9, 9, 2), dtype=np.float64)
        for row in range(9):
            for col in range(9):
                pts[row, col] = (col * 10.0, row * 10.0)
        corners = board_corners_from_intersections(pts)
        assert corners[0, 0] == 0.0
        assert corners[1, 0] == 80.0
        assert corners[2, 0] == 80.0
        assert corners[2, 1] == 80.0
