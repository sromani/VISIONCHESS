"""Class labels for square piece classification."""

from __future__ import annotations

from enum import IntEnum

CLASS_NAMES: tuple[str, ...] = (
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
)

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {idx: name for name, idx in CLASS_TO_IDX.items()}

FEN_SYMBOL_TO_LABEL: dict[str, str] = {
    "P": "white_pawn",
    "N": "white_knight",
    "B": "white_bishop",
    "R": "white_rook",
    "Q": "white_queen",
    "K": "white_king",
    "p": "black_pawn",
    "n": "black_knight",
    "b": "black_bishop",
    "r": "black_rook",
    "q": "black_queen",
    "k": "black_king",
}


class PieceLabel(IntEnum):
    EMPTY = 0
    WHITE_PAWN = 1
    WHITE_KNIGHT = 2
    WHITE_BISHOP = 3
    WHITE_ROOK = 4
    WHITE_QUEEN = 5
    WHITE_KING = 6
    BLACK_PAWN = 7
    BLACK_KNIGHT = 8
    BLACK_BISHOP = 9
    BLACK_ROOK = 10
    BLACK_QUEEN = 11
    BLACK_KING = 12


def validate_class_name(name: str) -> str:
    if name not in CLASS_TO_IDX:
        msg = f"Unknown class '{name}'. Expected one of: {CLASS_NAMES}"
        raise ValueError(msg)
    return name
