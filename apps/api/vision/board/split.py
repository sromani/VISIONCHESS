"""Split warped board into 64 named square crops with optional persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.debug_viz import (
    render_crop_montage,
    render_grid_debug_extreme,
    render_grid_overlay,
    square_polygons_metadata,
)
from vision.board.grid import BoardGridExtractor
from vision.board.grid_config import GridExtractorConfig
from vision.board.types import BoardGridResult, SquareCrop


@dataclass(frozen=True, slots=True)
class SavedSquare:
    name: str
    filename: str
    path: Path
    cell_bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "path": str(self.path),
            "cell_bbox": list(self.cell_bbox),
            "crop_bbox": list(self.crop_bbox),
        }


@dataclass(frozen=True, slots=True)
class BoardSplitResult:
    grid: BoardGridResult
    saved_squares: tuple[SavedSquare, ...]
    debug_overlay: NDArray[np.uint8]
    debug_montage: NDArray[np.uint8]
    debug_extreme: NDArray[np.uint8]
    squares_dir: Path

    def to_metadata(self) -> dict[str, Any]:
        return {
            "grid": self.grid.to_metadata(),
            "squares_dir": str(self.squares_dir),
            "square_polygons": square_polygons_metadata(self.grid),
            "squares": [sq.to_metadata() for sq in self.saved_squares],
        }


class BoardSquareSplitter:
    """Extract 64 squares from a warped board and save as ``a8.png`` … ``h1.png``."""

    def __init__(self, config: GridExtractorConfig | None = None) -> None:
        self._extractor = BoardGridExtractor(config)

    def split(
        self,
        warped_board: NDArray[np.uint8],
        output_dir: Path,
        *,
        board_size: int | None = None,
    ) -> BoardSplitResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        squares_dir = output_dir / "squares"
        squares_dir.mkdir(parents=True, exist_ok=True)

        grid = self._extractor.extract(warped_board, board_size=board_size, uniform=True)
        saved = self._save_crops(grid, squares_dir)
        debug_overlay = render_grid_overlay(warped_board, grid)
        debug_montage = render_crop_montage(grid)
        debug_extreme = render_grid_debug_extreme(warped_board, grid)

        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "grid_overlay.jpg"), debug_overlay)
        cv2.imwrite(str(debug_dir / "crop_montage.jpg"), debug_montage)
        cv2.imwrite(str(debug_dir / "grid_debug_extreme.jpg"), debug_extreme)

        return BoardSplitResult(
            grid=grid,
            saved_squares=tuple(saved),
            debug_overlay=debug_overlay,
            debug_montage=debug_montage,
            debug_extreme=debug_extreme,
            squares_dir=squares_dir,
        )

    def _save_crops(self, grid: BoardGridResult, squares_dir: Path) -> list[SavedSquare]:
        saved: list[SavedSquare] = []
        for sq in grid.flat:
            path = squares_dir / sq.filename
            if not cv2.imwrite(str(path), sq.image):
                msg = f"Failed to write square crop {sq.filename}"
                raise OSError(msg)
            saved.append(_to_saved(sq, path))
        return saved


def _to_saved(sq: SquareCrop, path: Path) -> SavedSquare:
    return SavedSquare(
        name=sq.square_name,
        filename=sq.filename,
        path=path,
        cell_bbox=sq.cell_bbox,
        crop_bbox=sq.bbox,
    )
