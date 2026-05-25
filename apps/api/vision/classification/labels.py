"""13-class piece labels — shared with ML training."""

from __future__ import annotations

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
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = dict(enumerate(CLASS_NAMES))

EMPTY_IDX = 0
WHITE_SLICE = slice(1, 7)
BLACK_SLICE = slice(7, 13)

PIECE_KINDS = ("pawn", "knight", "bishop", "rook", "queen", "king")
WHITE_LABELS = tuple(f"white_{k}" for k in PIECE_KINDS)
BLACK_LABELS = tuple(f"black_{k}" for k in PIECE_KINDS)

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
