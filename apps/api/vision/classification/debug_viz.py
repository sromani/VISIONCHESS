"""Classification debug visualizations — 8×8 grid with labels."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.types import BoardGridResult
from vision.classification.types import SquareClassification

LABEL_SHORT = {
    "empty": "·",
    "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
    "white_rook": "R", "white_queen": "Q", "white_king": "K",
    "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
    "black_rook": "r", "black_queen": "q", "black_king": "k",
}


def render_classification_grid(
    squares: list[SquareClassification] | tuple[SquareClassification, ...],
    cell_px: int = 96,
    gutter: int = 3,
) -> NDArray[np.uint8]:
    """Build 8×8 montage: crop + predicted label + confidence + empty flag."""
    label_h = 36
    tile = cell_px + label_h
    canvas_size = 8 * tile + 9 * gutter
    canvas = np.full((canvas_size, canvas_size, 3), 28, dtype=np.uint8)

    by_name = {sq.square_name: sq for sq in squares}

    for row in range(8):
        for col in range(8):
            rank = 8 - row
            file = chr(ord("a") + col)
            name = f"{file}{rank}"
            sq = by_name.get(name)
            if sq is None:
                continue

            ox = gutter + col * (tile + gutter)
            oy = gutter + row * (tile + gutter)

            cell = np.full((cell_px, cell_px, 3), 50 if (row + col) % 2 == 0 else 35, dtype=np.uint8)
            canvas[oy + label_h : oy + label_h + cell_px, ox : ox + cell_px] = cell

            if sq.occupied:
                sym = LABEL_SHORT.get(sq.label, "?")
                color = (220, 220, 220) if sq.label.startswith("white_") else (160, 160, 160)
                border = (80, 200, 80)
            else:
                sym = "·"
                color = (100, 100, 100)
                border = (80, 80, 120)

            cv2.rectangle(canvas, (ox, oy), (ox + cell_px, oy + label_h + cell_px), border, 1)

            cv2.putText(canvas, name, (ox + 4, oy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
            cv2.putText(canvas, sym, (ox + cell_px // 2 - 8, oy + label_h + cell_px // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            conf_text = f"{sq.confidence:.0%}" if sq.occupied else f"empty {sq.occupancy_score:.0%}"
            cv2.putText(canvas, conf_text, (ox + 4, oy + label_h + cell_px - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)

    return canvas


def render_classification_grid_with_crops(
    squares: list[SquareClassification] | tuple[SquareClassification, ...],
    dataset_grid: BoardGridResult,
    cell_px: int = 80,
    gutter: int = 2,
) -> NDArray[np.uint8]:
    """8×8 montage of dataset-quality crops exactly as the classifier sees them."""
    crop_map = {sq.square_name: sq for sq in dataset_grid.flat}
    by_name = {sq.square_name: sq for sq in squares}

    label_h = 32
    tile = cell_px + label_h
    canvas_size = 8 * tile + 9 * gutter
    canvas = np.full((canvas_size, canvas_size, 3), 28, dtype=np.uint8)

    for sq_crop in dataset_grid.flat:
        cls = by_name.get(sq_crop.square_name)
        if cls is None:
            continue

        ox = gutter + sq_crop.col * (tile + gutter)
        oy = gutter + sq_crop.row * (tile + gutter)

        resized = cv2.resize(sq_crop.image, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)
        canvas[oy + label_h : oy + label_h + cell_px, ox : ox + cell_px] = resized

        sym = LABEL_SHORT.get(cls.label if cls.occupied else "empty", "?")
        if cls.occupied:
            text_color = (80, 255, 80) if cls.label.startswith("white_") else (80, 180, 255)
        else:
            text_color = (120, 120, 180)

        cv2.rectangle(canvas, (ox, oy), (ox + cell_px - 1, oy + label_h + cell_px - 1), text_color, 1)
        cv2.putText(canvas, sq_crop.square_name, (ox + 3, oy + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
        line2 = f"{sym} {cls.confidence:.0%}" if cls.occupied else "empty"
        cv2.putText(canvas, line2, (ox + 3, oy + label_h + cell_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)

    return canvas
