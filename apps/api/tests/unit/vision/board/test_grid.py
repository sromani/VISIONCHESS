"""Unit tests for board grid extraction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.board.exceptions import InvalidGridError
from vision.board.grid import BoardGridExtractor
from vision.board.grid_lines import detect_grid_lines
from vision.board.types import SQUARES_PER_SIDE, TOTAL_SQUARES
from vision.pipeline import VisionPipeline


def _checkerboard(size: int = 800) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size / SQUARES_PER_SIDE
    for row in range(SQUARES_PER_SIDE):
        for col in range(SQUARES_PER_SIDE):
            color = 220 if (row + col) % 2 == 0 else 80
            y0 = int(round(row * cell))
            y1 = int(round((row + 1) * cell))
            x0 = int(round(col * cell))
            x1 = int(round((col + 1) * cell))
            board[y0:y1, x0:x1] = (color, color, color)
    return board


def _padded_checkerboard(outer: int = 800, inner: int = 720, offset: int = 40) -> np.ndarray:
    board = np.full((outer, outer, 3), 15, dtype=np.uint8)
    board[offset : offset + inner, offset : offset + inner] = _checkerboard(inner)
    return board


class TestGridLineDetection:
    def test_detects_lines_on_full_checkerboard(self) -> None:
        lines = detect_grid_lines(_checkerboard(800))
        assert len(lines.x_lines) == 9
        assert len(lines.y_lines) == 9
        assert lines.x_lines[0] < 15
        assert lines.x_lines[-1] > 785

    def test_detects_offset_board_not_image_origin(self) -> None:
        lines = detect_grid_lines(_padded_checkerboard())
        assert lines.x_lines[0] >= 30
        assert lines.y_lines[0] >= 30
        assert lines.x_lines[-1] <= 770
        assert lines.y_lines[-1] <= 770


class TestBoardGridExtractor:
    def test_extracts_64_squares(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard())
        assert len(grid.squares) == SQUARES_PER_SIDE
        assert all(len(row) == SQUARES_PER_SIDE for row in grid.squares)
        assert len(grid.flat) == TOTAL_SQUARES

    def test_square_dimensions_with_margin(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard(800))
        assert grid.board_size == 800
        assert 90 <= grid.square_size <= 110
        assert 6 <= grid.margin_px <= 10
        for sq in grid.flat:
            assert sq.image.shape[0] == sq.image.shape[1]
            assert 75 <= sq.image.shape[0] <= 95

    def test_square_naming(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard())
        assert grid.crop_at(0, 0).square_name == "a8"
        assert grid.crop_at(7, 7).square_name == "h1"

    def test_bboxes_tile_board(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard(800))
        for row in range(SQUARES_PER_SIDE):
            for col in range(SQUARES_PER_SIDE):
                sq = grid.crop_at(row, col)
                assert sq.image.shape[0] == sq.image.shape[1]
                assert sq.row == row
                assert sq.col == col
                assert len(sq.cell_quad) == 4
                assert len(sq.crop_quad) == 4

    def test_padded_board_aligns_to_content(self) -> None:
        grid = BoardGridExtractor().extract(_padded_checkerboard())
        a8 = grid.crop_at(0, 0)
        assert a8.cell_bbox[0] >= 30
        assert a8.cell_bbox[1] >= 30
        h1 = grid.crop_at(7, 7)
        assert h1.cell_bbox[2] <= 770
        assert h1.cell_bbox[3] <= 770

    def test_crops_are_uniform_color_on_checkerboard(self) -> None:
        """Each crop should be mostly one shade — not split across square boundaries."""
        grid = BoardGridExtractor().extract(_checkerboard(800))
        for sq in grid.flat:
            gray = cv2.cvtColor(sq.image, cv2.COLOR_BGR2GRAY)
            assert float(np.std(gray)) < 25.0

    def test_rejects_tiny_image(self) -> None:
        tiny = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(InvalidGridError):
            BoardGridExtractor().extract(tiny)

    def test_rgb_resize_for_ml(self) -> None:
        grid = BoardGridExtractor().extract_crops_rgb(_checkerboard(), output_size=64)
        for sq in grid.flat:
            assert sq.image.shape == (64, 64, 3)

    def test_grid_method_reported(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard())
        assert grid.grid_method in {
            "chessboard_corners",
            "gradient_profile",
            "hough_lines",
            "proportional_bounds",
        }


class TestVisionPipeline:
    def test_pipeline_produces_grid_from_skewed_photo(self) -> None:
        size = 600
        board = _checkerboard(size)
        src = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
        dst = np.float32([[70, 35], [530, 15], [490, 565], [25, 520]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        scene = cv2.warpPerspective(board, matrix, (600, 600), borderValue=(20, 20, 20))

        result = VisionPipeline().run(scene)
        assert result.detection.confidence > 0.25
        assert len(result.grid.flat) == TOTAL_SQUARES
        assert result.grid.grid_method == "dataset_export"
        assert result.grid.square_size == 64
        assert result.grid.board_size == 2048
        assert result.metadata["crop_quality"]["analysis_square_px"] >= 128
        assert result.metadata["classification"]["input_square_px"] >= 128
        assert result.timing.total_ms >= 0
