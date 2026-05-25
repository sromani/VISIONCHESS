"""Domain types for board detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

Point2D = tuple[float, float]

SQUARES_PER_SIDE = 8
TOTAL_SQUARES = SQUARES_PER_SIDE * SQUARES_PER_SIDE


@dataclass(frozen=True, slots=True)
class SquareCrop:
    """Single square extracted from a warped board.

    ``row`` 0 is the top visual rank (FEN rank 8); ``col`` 0 is file a.
    Crops are produced by perspective-warping ``crop_quad`` — never by
    axis-aligned slicing.
    """

    row: int
    col: int
    image: "NDArray[np.uint8]"
    bbox: tuple[int, int, int, int]  # axis-aligned envelope of crop_quad (metadata)
    cell_bbox: tuple[int, int, int, int]  # axis-aligned envelope of cell_quad (metadata)
    cell_quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    crop_quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]

    @property
    def rank(self) -> int:
        return SQUARES_PER_SIDE - self.row

    @property
    def file(self) -> str:
        return chr(ord("a") + self.col)

    @property
    def square_name(self) -> str:
        return f"{self.file}{self.rank}"

    @property
    def filename(self) -> str:
        return f"{self.square_name}.png"


BoardGrid = tuple[tuple[SquareCrop, ...], ...]


@dataclass(frozen=True, slots=True)
class BoardGridResult:
    """64 square crops from a perspective-corrected board."""

    squares: BoardGrid
    board_size: int
    square_size: int
    margin_px: int
    x_lines: tuple[float, ...] = ()
    y_lines: tuple[float, ...] = ()
    grid_method: str = "unknown"

    @property
    def flat(self) -> tuple[SquareCrop, ...]:
        return tuple(sq for row in self.squares for sq in row)

    def crop_at(self, row: int, col: int) -> SquareCrop:
        return self.squares[row][col]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "board_size": self.board_size,
            "square_size": self.square_size,
            "margin_px": self.margin_px,
            "grid_method": self.grid_method,
            "x_lines": [round(v, 2) for v in self.x_lines],
            "y_lines": [round(v, 2) for v in self.y_lines],
            "squares": [
                {
                    "row": sq.row,
                    "col": sq.col,
                    "name": sq.square_name,
                    "filename": sq.filename,
                    "cell_bbox": list(sq.cell_bbox),
                    "crop_bbox": list(sq.bbox),
                    "cell_quad": [list(pt) for pt in sq.cell_quad],
                    "crop_quad": [list(pt) for pt in sq.crop_quad],
                }
                for sq in self.flat
            ],
        }


@dataclass(frozen=True, slots=True)
class BoardDetectionResult:
    """Output of a successful board detection."""

    corners: NDArray[np.float32]  # shape (4, 2) ordered TL, TR, BR, BL
    homography: NDArray[np.float64]  # shape (3, 3)
    warped_board: NDArray[np.uint8]
    confidence: float
    original_size: tuple[int, int]  # (width, height)
    output_size: int

    @property
    def corners_list(self) -> list[Point2D]:
        return [(float(x), float(y)) for x, y in self.corners]

    def homography_list(self) -> list[list[float]]:
        return self.homography.astype(float).tolist()

    def to_metadata(self) -> dict[str, Any]:
        """JSON-serializable summary (excludes pixel buffers)."""
        return {
            "corners": self.corners_list,
            "homography": self.homography_list(),
            "confidence": self.confidence,
            "original_size": list(self.original_size),
            "output_size": self.output_size,
        }
