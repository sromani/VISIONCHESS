"""Split a warped board into an 8×8 grid of square crops."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.exceptions import InvalidGridError
from vision.board.grid_config import GridExtractorConfig
from vision.board.grid_intersections import (
    cell_quad,
    inset_quad,
    intersections_from_lines,
    quad_envelope,
    quad_to_tuple,
)
from vision.board.grid_lines import GridLineSet, detect_grid_lines
from vision.board.square_warp import quad_output_size, warp_quad_to_square
from vision.board.types import SQUARES_PER_SIDE, BoardGrid, BoardGridResult, SquareCrop

NUM_LINES = SQUARES_PER_SIDE + 1


class BoardGridExtractor:
    """Extract 64 square crops from a top-down warped board image."""

    def __init__(self, config: GridExtractorConfig | None = None) -> None:
        self._config = config or GridExtractorConfig()

    @property
    def config(self) -> GridExtractorConfig:
        return self._config

    def extract(
        self,
        warped_board: NDArray[np.uint8],
        board_size: int | None = None,
        *,
        uniform: bool = False,
    ) -> BoardGridResult:
        if uniform:
            return self.extract_uniform(warped_board, board_size)
        return self._extract_from_detected_lines(warped_board, board_size)

    def extract_uniform(
        self,
        warped_board: NDArray[np.uint8],
        board_size: int | None = None,
    ) -> BoardGridResult:
        """Uniform 8×8 split — use only after mesh rectification."""
        image = _ensure_bgr(warped_board)
        height, width = image.shape[:2]
        _validate_image(image)

        size = board_size or min(height, width)
        cell = size // SQUARES_PER_SIDE
        if cell < self._config.min_crop_px:
            msg = f"Board too small for uniform split: cell={cell}px"
            raise InvalidGridError(msg)

        used = cell * SQUARES_PER_SIDE
        if height != used or width != used:
            y0 = max(0, (height - used) // 2)
            x0 = max(0, (width - used) // 2)
            image = image[y0 : y0 + used, x0 : x0 + used]

        step = float(cell)
        x_lines = tuple(i * step for i in range(NUM_LINES))
        y_lines = tuple(i * step for i in range(NUM_LINES))
        intersections = intersections_from_lines(x_lines, y_lines)
        margin_px = max(1, int(round(cell * self._config.margin_ratio)))

        rows: list[tuple[SquareCrop, ...]] = []
        for row in range(SQUARES_PER_SIDE):
            cells: list[SquareCrop] = []
            for col in range(SQUARES_PER_SIDE):
                y0, y1 = row * cell, (row + 1) * cell
                x0, x1 = col * cell, (col + 1) * cell
                cell_img = image[y0:y1, x0:x1]
                crop_img = cell_img[margin_px : cell - margin_px, margin_px : cell - margin_px]
                if crop_img.shape[0] < self._config.min_crop_px:
                    msg = f"Uniform cell ({row},{col}) too small after margin"
                    raise InvalidGridError(msg)

                cell_q = cell_quad(intersections, row, col)
                crop_q = inset_quad(cell_q, self._config.margin_ratio)
                cells.append(
                    SquareCrop(
                        row=row,
                        col=col,
                        image=crop_img.copy(),
                        bbox=quad_envelope(crop_q),
                        cell_bbox=quad_envelope(cell_q),
                        cell_quad=quad_to_tuple(cell_q),  # type: ignore[arg-type]
                        crop_quad=quad_to_tuple(crop_q),  # type: ignore[arg-type]
                    )
                )
            rows.append(tuple(cells))

        return BoardGridResult(
            squares=tuple(rows),
            board_size=used,
            square_size=cell - 2 * margin_px,
            margin_px=margin_px,
            x_lines=x_lines,
            y_lines=y_lines,
            grid_method="uniform_mesh",
        )

    def _extract_from_detected_lines(
        self,
        warped_board: NDArray[np.uint8],
        board_size: int | None = None,
    ) -> BoardGridResult:
        image = _ensure_bgr(warped_board)
        height, width = image.shape[:2]
        _validate_image(image)

        line_set = detect_grid_lines(image, self._config)
        rows = _crop_squares(image, line_set, self._config)

        avg_cell = int(round(line_set.avg_cell_size))
        avg_margin = _average_margin(rows)

        return BoardGridResult(
            squares=tuple(rows),
            board_size=board_size or max(height, width),
            square_size=avg_cell,
            margin_px=avg_margin,
            x_lines=line_set.x_lines,
            y_lines=line_set.y_lines,
            grid_method=line_set.method,
        )

    def extract_crops_rgb(
        self,
        warped_board: NDArray[np.uint8],
        board_size: int | None = None,
        output_size: int = 64,
    ) -> BoardGridResult:
        """Extract squares and resize to ``output_size`` in RGB for ML classifiers."""
        grid = self.extract(warped_board, board_size)
        resized_rows: list[tuple[SquareCrop, ...]] = []

        for row in grid.squares:
            cells: list[SquareCrop] = []
            for sq in row:
                rgb = cv2.cvtColor(sq.image, cv2.COLOR_BGR2RGB)
                if output_size > 0 and (rgb.shape[0] != output_size or rgb.shape[1] != output_size):
                    rgb = cv2.resize(rgb, (output_size, output_size), interpolation=cv2.INTER_AREA)
                cells.append(
                    SquareCrop(
                        row=sq.row,
                        col=sq.col,
                        image=rgb,
                        bbox=sq.bbox,
                        cell_bbox=sq.cell_bbox,
                        cell_quad=sq.cell_quad,
                        crop_quad=sq.crop_quad,
                    )
                )
            resized_rows.append(tuple(cells))

        return BoardGridResult(
            squares=tuple(resized_rows),
            board_size=grid.board_size,
            square_size=grid.square_size,
            margin_px=grid.margin_px,
            x_lines=grid.x_lines,
            y_lines=grid.y_lines,
            grid_method=grid.grid_method,
        )


def _crop_squares(
    image: NDArray[np.uint8],
    line_set: GridLineSet,
    config: GridExtractorConfig,
) -> list[tuple[SquareCrop, ...]]:
    intersections = line_set.intersection_array
    rows: list[tuple[SquareCrop, ...]] = []

    for row in range(SQUARES_PER_SIDE):
        cells: list[SquareCrop] = []
        for col in range(SQUARES_PER_SIDE):
            cell_q = cell_quad(intersections, row, col)
            crop_q = inset_quad(cell_q, config.margin_ratio)
            output_size = quad_output_size(crop_q)

            if output_size < config.min_crop_px:
                msg = f"Grid cell ({row},{col}) too small after warp: {output_size}px"
                raise InvalidGridError(msg)

            crop = warp_quad_to_square(image, crop_q, output_size)
            cell_tuple = quad_to_tuple(cell_q)
            crop_tuple = quad_to_tuple(crop_q)

            cells.append(
                SquareCrop(
                    row=row,
                    col=col,
                    image=crop,
                    bbox=quad_envelope(crop_q),
                    cell_bbox=quad_envelope(cell_q),
                    cell_quad=cell_tuple,  # type: ignore[arg-type]
                    crop_quad=crop_tuple,  # type: ignore[arg-type]
                )
            )
        rows.append(tuple(cells))

    return rows


def _average_margin(rows: list[tuple[SquareCrop, ...]]) -> int:
    if not rows:
        return 0

    edges: list[float] = []
    for row in rows:
        for sq in row:
            cell = np.array(sq.cell_quad, dtype=np.float64)
            crop = np.array(sq.crop_quad, dtype=np.float64)
            cell_mean = float(np.mean([
                np.linalg.norm(cell[1] - cell[0]),
                np.linalg.norm(cell[2] - cell[3]),
            ]))
            crop_mean = float(np.mean([
                np.linalg.norm(crop[1] - crop[0]),
                np.linalg.norm(crop[2] - crop[3]),
            ]))
            if cell_mean > 0:
                edges.append((cell_mean - crop_mean) / 2.0)

    return int(round(float(np.mean(edges)))) if edges else 0


def _ensure_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    msg = f"Expected H×W or H×W×3 image, got shape {image.shape}"
    raise InvalidGridError(msg)


def _validate_image(image: NDArray[np.uint8]) -> None:
    height, width = image.shape[:2]
    min_dim = SQUARES_PER_SIDE * 16
    if height < min_dim or width < min_dim:
        msg = f"Image {width}×{height} too small for 8×8 extraction (min {min_dim}px)"
        raise InvalidGridError(msg)
