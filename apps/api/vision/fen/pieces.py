"""Piece label mapping and FEN symbols."""

from __future__ import annotations

from enum import StrEnum


class PieceLabel(StrEnum):
    EMPTY = "empty"
    WHITE_PAWN = "white_pawn"
    WHITE_KNIGHT = "white_knight"
    WHITE_BISHOP = "white_bishop"
    WHITE_ROOK = "white_rook"
    WHITE_QUEEN = "white_queen"
    WHITE_KING = "white_king"
    BLACK_PAWN = "black_pawn"
    BLACK_KNIGHT = "black_knight"
    BLACK_BISHOP = "black_bishop"
    BLACK_ROOK = "black_rook"
    BLACK_QUEEN = "black_queen"
    BLACK_KING = "black_king"


LABEL_TO_FEN: dict[str, str] = {
    PieceLabel.WHITE_PAWN: "P",
    PieceLabel.WHITE_KNIGHT: "N",
    PieceLabel.WHITE_BISHOP: "B",
    PieceLabel.WHITE_ROOK: "R",
    PieceLabel.WHITE_QUEEN: "Q",
    PieceLabel.WHITE_KING: "K",
    PieceLabel.BLACK_PAWN: "p",
    PieceLabel.BLACK_KNIGHT: "n",
    PieceLabel.BLACK_BISHOP: "b",
    PieceLabel.BLACK_ROOK: "r",
    PieceLabel.BLACK_QUEEN: "q",
    PieceLabel.BLACK_KING: "k",
}

FEN_TO_LABEL: dict[str, str] = {symbol: label.value for label, symbol in LABEL_TO_FEN.items()}

KING_LABELS = frozenset({PieceLabel.WHITE_KING, PieceLabel.BLACK_KING})
PAWN_LABELS = frozenset({PieceLabel.WHITE_PAWN, PieceLabel.BLACK_PAWN})


def is_known_label(label: str) -> bool:
    if label == PieceLabel.EMPTY:
        return True
    return label in LABEL_TO_FEN


def to_fen_symbol(label: str) -> str | None:
    if label == PieceLabel.EMPTY:
        return None
    return LABEL_TO_FEN.get(label)
