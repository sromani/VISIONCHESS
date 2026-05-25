"""Tests for chessboard_detector module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.board.exceptions import BoardNotFoundError
from vision.chessboard_detector import ChessboardDetector, ChessboardDetectorConfig


def _checkerboard(size: int = 800) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 70
            board[row * sq : (row + 1) * sq, col * sq : (col + 1) * sq] = (color, color, color)
    return board


def _skewed_scene(size: int = 600) -> np.ndarray:
    board = _checkerboard(size)
    src = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    dst = np.float32([[75, 40], [525, 20], [485, 560], [35, 525]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board, matrix, (600, 600), borderValue=(25, 25, 25))


class TestChessboardDetector:
    def test_detects_skewed_board(self) -> None:
        result = ChessboardDetector().detect(_skewed_scene())
        assert result.confidence > 0.3
        assert result.warped_image.shape == (800, 800, 3)
        assert len(result.corners_list) == 4
        assert result.original_width == 600
        assert result.rectification_method.startswith("mesh_")

    def test_respects_output_size(self) -> None:
        config = ChessboardDetectorConfig(output_size=512)
        result = ChessboardDetector(config).detect(_skewed_scene())
        assert result.output_width == 512
        assert result.warped_image.shape[:2] == (512, 512)

    def test_handles_large_input(self) -> None:
        large = cv2.resize(_skewed_scene(), (2400, 1800))
        result = ChessboardDetector().detect(large)
        assert result.warped_image.shape == (800, 800, 3)

    def test_metadata_serializable(self) -> None:
        result = ChessboardDetector().detect(_skewed_scene())
        meta = result.to_metadata()
        assert meta["original_width"] == 600
        assert len(meta["corners"]) == 4
        assert len(meta["homography"]) == 3

    def test_raises_on_blank_image(self) -> None:
        blank = np.zeros((400, 400, 3), dtype=np.uint8)
        with pytest.raises(BoardNotFoundError):
            ChessboardDetector().detect(blank)

    def test_raises_on_empty_image(self) -> None:
        with pytest.raises(BoardNotFoundError):
            ChessboardDetector().detect(np.array([], dtype=np.uint8))
