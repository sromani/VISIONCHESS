"""Tests for production ScanPipeline."""

from __future__ import annotations

import cv2
import numpy as np

from vision.scanner import ScanPipeline


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


class TestScanPipeline:
    def test_full_grid_pipeline(self) -> None:
        result = ScanPipeline().run(_skewed_scene())
        assert result.confidence > 0.25
        assert result.warped_board.shape == (800, 800, 3)
        assert result.classification.fen
        assert len(result.classification.squares) == 64
        assert result.metadata.get("pipeline") == "geometry_first_scanner_v2"
        assert "observed_grid" in result.metadata
        assert "canonical_grid" in result.metadata

    def test_debug_frames(self) -> None:
        result = ScanPipeline().run(_skewed_scene())
        assert "detected_lines" in result.debug_jpegs
        assert "rectified_board" in result.debug_jpegs
        assert "occupancy" in result.debug_jpegs
