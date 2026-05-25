"""Tests for board square splitting and debug visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision.board.grid import BoardGridExtractor
from vision.board.debug_viz import render_grid_debug_extreme, square_polygons_metadata
from vision.board.split import BoardSquareSplitter


def _checkerboard(size: int = 800) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size / 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 70
            y0 = int(round(row * cell))
            y1 = int(round((row + 1) * cell))
            x0 = int(round(col * cell))
            x1 = int(round((col + 1) * cell))
            board[y0:y1, x0:x1] = (color, color, color)
    return board


class TestBoardSquareSplitter:
    def test_saves_64_named_crops(self, tmp_path: Path) -> None:
        result = BoardSquareSplitter().split(_checkerboard(), tmp_path)
        assert len(result.saved_squares) == 64
        names = {sq.name for sq in result.saved_squares}
        assert names == {f"{f}{r}" for f in "abcdefgh" for r in range(1, 9)}
        assert (result.squares_dir / "a8.png").exists()
        assert (result.squares_dir / "h1.png").exists()

    def test_cell_coordinates_on_full_board(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard(800))
        a8 = grid.crop_at(0, 0)
        assert a8.cell_bbox[0] < 15
        assert a8.cell_bbox[1] < 15
        assert a8.square_name == "a8"

        h1 = grid.crop_at(7, 7)
        assert h1.cell_bbox[2] > 785
        assert h1.cell_bbox[3] > 785
        assert h1.square_name == "h1"

    def test_debug_artifacts_written(self, tmp_path: Path) -> None:
        result = BoardSquareSplitter().split(_checkerboard(), tmp_path)
        assert result.debug_overlay.shape[0] == 800
        assert result.debug_montage.shape[0] > 0
        assert result.debug_extreme.shape[0] == 800
        assert (tmp_path / "debug" / "grid_overlay.jpg").exists()
        assert (tmp_path / "debug" / "crop_montage.jpg").exists()
        assert (tmp_path / "debug" / "grid_debug_extreme.jpg").exists()

    def test_extreme_debug_has_64_polygons(self, tmp_path: Path) -> None:
        result = BoardSquareSplitter().split(_checkerboard(), tmp_path)
        polygons = square_polygons_metadata(result.grid)
        assert len(polygons) == 64
        assert polygons[0]["name"] == "a8"
        assert len(polygons[0]["polygon"]) == 4
        assert "center" in polygons[0]

    def test_extreme_debug_renders(self) -> None:
        grid = BoardGridExtractor().extract(_checkerboard(800))
        extreme = render_grid_debug_extreme(_checkerboard(800), grid)
        assert extreme.shape == (800, 800, 3)
        assert extreme.sum() > 0

    def test_metadata_lists_all_squares(self, tmp_path: Path) -> None:
        result = BoardSquareSplitter().split(_checkerboard(), tmp_path)
        meta = result.to_metadata()
        assert len(meta["squares"]) == 64
        first = meta["squares"][0]
        assert "cell_bbox" in first
        assert "crop_bbox" in first
        assert first["filename"].endswith(".png")
        assert "grid_method" in meta["grid"]
