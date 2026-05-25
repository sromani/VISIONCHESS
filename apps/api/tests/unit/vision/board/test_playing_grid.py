"""Tests for PlayingGrid geometric model."""

from __future__ import annotations

import cv2
import numpy as np

from vision.board.grid_solver import solve_canonical
from vision.board.playing_grid import GridConstraints, PlayingGrid, measure_constraints


def _skewed_intersections(size: int = 600) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size / 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 70
            y0, y1 = int(row * cell), int((row + 1) * cell)
            x0, x1 = int(col * cell), int((col + 1) * cell)
            board[y0:y1, x0:x1] = color
    src = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    dst = np.float32([[75, 40], [525, 20], [485, 560], [35, 525]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, matrix, (600, 600), borderValue=(25, 25, 25))

    from vision.board.grid_homography import detect_grid_rectification

    grid = detect_grid_rectification(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    return grid.intersections


class TestPlayingGrid:
    def test_canonical_has_zero_constraints(self) -> None:
        grid = PlayingGrid.canonical(800.0)
        m = grid.metrics()
        assert m.column_width_cv == 0.0
        assert m.row_height_cv == 0.0
        assert m.orthogonality_error_deg == 0.0
        assert m.row_parallelism_deg == 0.0
        assert m.col_parallelism_deg == 0.0
        assert grid.satisfies()

    def test_skewed_observed_violates_constraints(self) -> None:
        observed = PlayingGrid.from_intersections(_skewed_intersections())
        assert observed.metrics().max_error > 0.001
        assert not observed.satisfies(GridConstraints(max_orthogonality_deg=2.0))

    def test_solve_produces_canonical(self) -> None:
        observed = PlayingGrid.from_intersections(_skewed_intersections())
        solve = solve_canonical(observed, 800)
        assert solve.canonical_metrics.column_width_cv == 0.0
        assert solve.constraints_satisfied
        assert len(solve.observed.all_cells()) == 64

    def test_cell_geometry_from_lattice(self) -> None:
        grid = PlayingGrid.canonical(800.0)
        cell = grid.cell(0, 0)
        assert cell.square_name == "a8"
        assert abs(cell.width - 99.875) < 0.01
        assert abs(cell.height - 99.875) < 0.01
