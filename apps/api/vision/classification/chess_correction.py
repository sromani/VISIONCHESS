"""Chess-aware post-correction of square classifications."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from vision.classification.labels import CLASS_NAMES
from vision.classification.types import SquareClassification

MAX_PER_COLOR: dict[str, int] = {
    "pawn": 8,
    "knight": 2,
    "bishop": 2,
    "rook": 2,
    "queen": 1,
    "king": 1,
}


def apply_chess_constraints(
    squares: list[SquareClassification],
) -> list[SquareClassification]:
    """Penalize impossible piece counts and missing kings."""
    corrected = list(squares)
    corrected = _fix_excess_pieces(corrected)
    corrected = _ensure_kings(corrected)
    return corrected


def _piece_kind(label: str) -> str | None:
    if label == "empty" or "_" not in label:
        return None
    return label.split("_", 1)[1]


def _color(label: str) -> str | None:
    if label == "empty" or "_" not in label:
        return None
    return label.split("_", 1)[0]


def _fix_excess_pieces(squares: list[SquareClassification]) -> list[SquareClassification]:
    out = list(squares)
    for color in ("white", "black"):
        for kind, maximum in MAX_PER_COLOR.items():
            label = f"{color}_{kind}"
            indices = [i for i, sq in enumerate(out) if sq.label == label]
            if len(indices) <= maximum:
                continue
            indices.sort(key=lambda i: out[i].confidence)
            for i in indices[:-maximum]:
                sq = out[i]
                out[i] = replace(
                    sq,
                    label="empty",
                    occupied=False,
                    confidence=sq.confidence * 0.35,
                    empty_reason="excess_piece_count",
                )
    return out


def _ensure_kings(squares: list[SquareClassification]) -> list[SquareClassification]:
    out = list(squares)
    for color in ("white", "black"):
        king_label = f"{color}_king"
        kings = [i for i, sq in enumerate(out) if sq.label == king_label]
        if len(kings) == 1:
            continue
        if len(kings) > 1:
            kings.sort(key=lambda i: out[i].confidence, reverse=True)
            for i in kings[1:]:
                sq = out[i]
                out[i] = replace(sq, label=f"{color}_queen", confidence=sq.confidence * 0.6)
            continue
        candidates = [
            i
            for i, sq in enumerate(out)
            if sq.occupied and _color(sq.label) == color and _piece_kind(sq.label) != "pawn"
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: out[i].confidence)
        sq = out[best]
        out[best] = replace(sq, label=king_label, confidence=sq.confidence * 0.75)
    return out


def board_piece_histogram(squares: list[SquareClassification]) -> dict[str, int]:
    counts = Counter(sq.label for sq in squares if sq.occupied)
    return {name: counts.get(name, 0) for name in CLASS_NAMES if name != "empty"}
