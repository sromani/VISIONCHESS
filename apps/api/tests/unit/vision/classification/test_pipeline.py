"""Unit tests for classification pipeline."""

from __future__ import annotations

import cv2
import numpy as np

from vision.occupancy.detector import detect_square
from vision.classification.orientation import flip_grid_vertical, orientation_layout_score
from vision.classification.pipeline import ClassificationPipeline
from vision.fen.types import SquarePrediction, grid_from_labels


def _checkerboard(size: int = 400) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 70
            board[row * sq : (row + 1) * sq, col * sq : (col + 1) * sq] = (color, color, color)
    return board


def _starting_position_board(size: int = 400) -> np.ndarray:
    board = _checkerboard(size)
    sq = size // 8
    # white pawns rank 2
    for col in range(8):
        y0, y1 = 6 * sq, 7 * sq
        x0, x1 = col * sq, (col + 1) * sq
        board[y0:y1, x0:x1] = (240, 240, 240)
    # black pawns rank 7
    for col in range(8):
        y0, y1 = 1 * sq, 2 * sq
        x0, x1 = col * sq, (col + 1) * sq
        board[y0:y1, x0:x1] = (30, 30, 30)
    # white king back rank
    board[7 * sq : 8 * sq, 4 * sq : 5 * sq] = (250, 250, 250)
    # black king back rank
    board[0:sq, 4 * sq : 5 * sq] = (20, 20, 20)
    return board


class TestEmptyDetection:
    def test_empty_square_low_variance(self) -> None:
        crop = np.full((64, 64, 3), 120, dtype=np.uint8)
        result = detect_square(crop, 0, 0)
        assert result.probability < 0.35
        assert not result.occupied

    def test_occupied_square_has_edges(self) -> None:
        crop = np.full((64, 64, 3), 120, dtype=np.uint8)
        cv2.circle(crop, (32, 32), 18, (240, 240, 240), -1)
        result = detect_square(crop, 6, 4, "e2")
        assert result.probability > 0.20
        assert not result.occupied


class TestOrientation:
    def test_flip_vertical_inverts_layout_score(self) -> None:
        labels = [
            ["black_king"] + ["empty"] * 7,
            ["empty"] * 8,
            ["empty"] * 8,
            ["empty"] * 8,
            ["empty"] * 8,
            ["empty"] * 8,
            ["empty"] * 8,
            ["white_king"] + ["empty"] * 7,
        ]
        grid = grid_from_labels(labels)
        normal = orientation_layout_score(grid)
        flipped = orientation_layout_score(flip_grid_vertical(grid))
        assert normal > flipped


class TestClassificationPipeline:
    def test_pipeline_returns_fen(self) -> None:
        board = _starting_position_board()
        result = ClassificationPipeline().run(board, board_size=400)
        assert result.fen
        assert len(result.squares) == 64
        assert result.orientation in ("normal", "flipped")

    def test_pipeline_has_candidates(self) -> None:
        board = _starting_position_board()
        result = ClassificationPipeline().run(board, board_size=400)
        assert len(result.candidates) == 2

    def test_pipeline_produces_dataset_grid(self) -> None:
        board = _starting_position_board()
        result = ClassificationPipeline().run(board, board_size=400)
        assert result.dataset_grid is not None
        assert len(result.dataset_grid.flat) == 64
        assert result.dataset_grid.flat[0].image.shape == (64, 64, 3)
