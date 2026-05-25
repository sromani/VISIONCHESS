"""Board-level validation rules."""

from __future__ import annotations

import chess

from vision.fen.pieces import KING_LABELS, PAWN_LABELS, PieceLabel
from vision.fen.types import FenIssue, FenIssueDetail, Grid8x8


def validate_kings(grid: Grid8x8) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    white: list[tuple[int, int]] = []
    black: list[tuple[int, int]] = []
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell.label == PieceLabel.WHITE_KING:
                white.append((row_idx, col_idx))
            elif cell.label == PieceLabel.BLACK_KING:
                black.append((row_idx, col_idx))
    return white, black


def validate_pawn_ranks(grid: Grid8x8) -> list[tuple[int, int]]:
    invalid: list[tuple[int, int]] = []
    for row_idx, row in enumerate(grid):
        if row_idx not in (0, 7):
            continue
        for col_idx, cell in enumerate(row):
            if cell.label in PAWN_LABELS:
                invalid.append((row_idx, col_idx))
    return invalid


def count_pieces(grid: Grid8x8) -> int:
    return sum(1 for row in grid for cell in row if cell.label != PieceLabel.EMPTY)


def validate_with_python_chess(fen: str) -> FenIssueDetail | None:
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return FenIssueDetail(
            code=FenIssue.ILLEGAL_POSITION,
            message=str(exc),
            severity=0.5,
        )

    if not board.is_valid():
        return FenIssueDetail(
            code=FenIssue.ILLEGAL_POSITION,
            message="Position rejected by python-chess validators",
            severity=0.4,
        )
    return None
