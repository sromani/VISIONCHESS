"""Sample pixel data through geometric cells — image is input, grid is truth."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.grid_intersections import inset_quad, quad_envelope, quad_to_tuple
from vision.board.playing_grid import PlayingGrid
from vision.board.square_warp import quad_output_size, warp_quad_to_square
from vision.board.types import SQUARES_PER_SIDE, BoardGridResult, SquareCrop


def sample_cell(
    image: NDArray[np.uint8],
    grid: PlayingGrid,
    row: int,
    col: int,
    *,
    margin_ratio: float = 0.08,
    min_crop_px: int = 8,
) -> SquareCrop:
    """Extract one square by warping the cell polygon — never axis-aligned bbox."""
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cell_geo = grid.cell(row, col)
    crop_quad = inset_quad(cell_geo.quad, margin_ratio)
    output_size = max(min_crop_px, quad_output_size(crop_quad))
    crop = warp_quad_to_square(bgr, crop_quad, output_size)

    return SquareCrop(
        row=row,
        col=col,
        image=crop,
        bbox=quad_envelope(crop_quad),
        cell_bbox=quad_envelope(cell_geo.quad),
        cell_quad=quad_to_tuple(cell_geo.quad),  # type: ignore[arg-type]
        crop_quad=quad_to_tuple(crop_quad),  # type: ignore[arg-type]
    )


def sample_all_cells(
    image: NDArray[np.uint8],
    grid: PlayingGrid,
    *,
    margin_ratio: float = 0.08,
    min_crop_px: int = 8,
    method: str = "geometry_sample",
) -> BoardGridResult:
    """Sample all 64 cells through the lattice geometry."""
    rows: list[tuple[SquareCrop, ...]] = []
    sizes: list[int] = []

    for row in range(SQUARES_PER_SIDE):
        cells: list[SquareCrop] = []
        for col in range(SQUARES_PER_SIDE):
            sq = sample_cell(
                image,
                grid,
                row,
                col,
                margin_ratio=margin_ratio,
                min_crop_px=min_crop_px,
            )
            cells.append(sq)
            sizes.append(sq.image.shape[0])
        rows.append(tuple(cells))

    avg = int(round(float(np.mean(sizes)))) if sizes else min_crop_px
    return BoardGridResult(
        squares=tuple(rows),
        board_size=int(grid.intersections[-1, -1].max()),
        square_size=avg,
        margin_px=int(round(avg * margin_ratio)),
        grid_method=method,
    )
