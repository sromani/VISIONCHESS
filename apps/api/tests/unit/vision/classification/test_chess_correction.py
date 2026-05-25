"""Tests for chess-aware post-correction."""

from __future__ import annotations

from vision.classification.chess_correction import apply_chess_constraints
from vision.classification.types import SquareClassification


def _sq(name: str, label: str, conf: float) -> SquareClassification:
    row = ord(name[1]) - ord("1")
    col = ord(name[0]) - ord("a")
    occupied = label != "empty"
    return SquareClassification(
        row=row,
        col=col,
        square_name=name,
        label=label,
        confidence=conf,
        occupied=occupied,
        occupancy_score=conf if occupied else 1.0 - conf,
    )


class TestChessCorrection:
    def test_excess_pawns_demoted_to_empty(self) -> None:
        squares = [_sq(f"a{i}", "white_pawn", 0.9 - i * 0.01) for i in range(1, 11)]
        squares += [_sq("e1", "white_king", 0.95)]
        corrected = apply_chess_constraints(squares)
        pawn_count = sum(1 for sq in corrected if sq.label == "white_pawn")
        assert pawn_count == 8

    def test_missing_king_promoted_from_best_piece(self) -> None:
        squares = [
            _sq("e1", "white_queen", 0.92),
            _sq("d1", "white_rook", 0.88),
            _sq("e8", "black_king", 0.95),
        ]
        corrected = apply_chess_constraints(squares)
        kings = [sq for sq in corrected if sq.label == "white_king"]
        assert len(kings) == 1
        assert kings[0].square_name == "e1"
