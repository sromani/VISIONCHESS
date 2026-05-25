"""Tests for grid-constrained mesh rectification."""

from __future__ import annotations

import cv2
import numpy as np

from vision.board.grid_homography import detect_grid_rectification
from vision.board.grid_lines import detect_grid_lines
from vision.board.mesh_rectification import measure_mesh_quality, rectify_board_mesh


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


class TestMeshRectification:
    def test_rectified_mesh_is_perfectly_uniform(self) -> None:
        scene = _skewed_scene()
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        grid = detect_grid_rectification(gray)
        source_stats = measure_mesh_quality(grid.intersections)
        assert source_stats.max_error > 0.001

        result = rectify_board_mesh(scene, grid.intersections, 800)
        stats = result.rectified_stats
        assert stats.column_width_cv == 0.0
        assert stats.row_height_cv == 0.0
        assert stats.orthogonality_error_deg == 0.0

    def test_warp_produces_parallel_uniform_grid(self) -> None:
        scene = _skewed_scene()
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        grid = detect_grid_rectification(gray)
        warped = rectify_board_mesh(scene, grid.intersections, 800).warped_image

        lines = detect_grid_lines(warped)
        x_sp = np.diff(lines.x_lines)
        y_sp = np.diff(lines.y_lines)
        assert float(np.std(x_sp) / np.mean(x_sp)) < 0.02
        assert float(np.std(y_sp) / np.mean(y_sp)) < 0.02

    def test_output_cells_are_equal_size(self) -> None:
        scene = _skewed_scene()
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        grid = detect_grid_rectification(gray)
        result = rectify_board_mesh(scene, grid.intersections, 800)
        assert result.cell_size == 100
        assert result.warped_image.shape == (800, 800, 3)
