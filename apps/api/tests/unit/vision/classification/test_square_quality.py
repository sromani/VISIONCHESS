"""Tests for dataset-quality square normalization."""

from __future__ import annotations

import cv2
import numpy as np

from vision.board.grid import BoardGridExtractor
from vision.classification.square_quality import (
    DatasetSquareConfig,
    normalize_grid_for_dataset,
    normalize_square_for_dataset,
)


def _checkerboard(size: int = 800) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size / 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 80
            y0, y1 = int(row * cell), int((row + 1) * cell)
            x0, x1 = int(col * cell), int((col + 1) * cell)
            board[y0:y1, x0:x1] = (color, color, color)
    return board


class TestDatasetSquareQuality:
    def test_output_is_fixed_size(self) -> None:
        raw = np.random.randint(0, 255, (73, 73, 3), dtype=np.uint8)
        out = normalize_square_for_dataset(raw, DatasetSquareConfig(output_size=64))
        assert out.shape == (64, 64, 3)

    def test_uniform_square_stays_uniform(self) -> None:
        raw = np.full((90, 90, 3), 180, dtype=np.uint8)
        out = normalize_square_for_dataset(raw)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        assert float(np.std(gray)) < 20.0

    def test_grid_normalization_preserves_count(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard())
        normalized = normalize_grid_for_dataset(grid)
        assert len(normalized.flat) == 64
        for sq in normalized.flat:
            assert sq.image.shape == (64, 64, 3)

    def test_checkerboard_cells_low_variance_after_pipeline(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard())
        normalized = normalize_grid_for_dataset(grid)
        for sq in normalized.flat:
            gray = cv2.cvtColor(sq.image, cv2.COLOR_BGR2GRAY)
            assert float(np.std(gray)) < 30.0
