"""Tests for context-aware crop extraction."""

from __future__ import annotations

import numpy as np

from vision.board.types import BoardGridResult, SquareCrop
from vision.classification.context_crop import (
    build_context_grid,
    compare_crop_predictions,
    extract_context_crop,
)


def _make_grid(cell: int = 64) -> BoardGridResult:
    quad = ((0.0, 0.0), (float(cell), 0.0), (float(cell), float(cell)), (0.0, float(cell)))
    rows = []
    for r in range(8):
        cells = []
        for c in range(8):
            img = np.full((cell, cell, 3), (r * 30 + c * 3, 100, 150), dtype=np.uint8)
            cells.append(
                SquareCrop(
                    row=r,
                    col=c,
                    image=img,
                    bbox=(0, 0, cell, cell),
                    cell_bbox=(c * cell, r * cell, (c + 1) * cell, (r + 1) * cell),
                    cell_quad=quad,
                    crop_quad=quad,
                )
            )
        rows.append(tuple(cells))
    return BoardGridResult(squares=tuple(rows), board_size=512, square_size=cell, margin_px=0)


def test_context_crop_larger_than_tight() -> None:
    board = np.zeros((512, 512, 3), dtype=np.uint8)
    tight = _make_grid().crop_at(3, 4).image.shape[0]
    ctx = extract_context_crop(board, 3, 4, scale=1.5)
    assert ctx.shape[0] >= tight


def test_build_context_grid_aligns_with_reference() -> None:
    board = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    ref = _make_grid()
    ctx_grid = build_context_grid(board, ref, scale=1.5)
    assert len(list(ctx_grid.flat)) == 64
    assert ctx_grid.crop_at(0, 0).square_name == "a8"
    assert ctx_grid.square_size >= ref.square_size


def test_compare_crop_predictions_finds_disagreements() -> None:
    tight = {"e4": {"label": "black_pawn", "confidence": 0.55}}
    context = {"e4": {"label": "white_queen", "confidence": 0.82}}
    report = compare_crop_predictions(tight, context)
    assert report["disagreement_count"] == 1
    assert report["context_higher_confidence"] == ["e4"]
