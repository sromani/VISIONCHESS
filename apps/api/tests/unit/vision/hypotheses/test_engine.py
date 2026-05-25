"""Tests for board hypothesis engine."""

from __future__ import annotations

from vision.classification.types import SquareClassification
from vision.hypotheses.engine import generate_board_hypotheses


def _sq(name: str, row: int, col: int, label: str, conf: float, occ: float) -> SquareClassification:
    return SquareClassification(
        row=row,
        col=col,
        square_name=name,
        label=label,
        confidence=conf,
        occupied=label != "empty",
        occupancy_score=occ,
    )


def _starting_finalized() -> list[SquareClassification]:
    squares: list[SquareClassification] = []
    for row in range(8):
        for col in range(8):
            name = f"{chr(ord('a') + col)}{8 - row}"
            label = "empty"
            conf = 0.9
            occ = 0.1
            if row == 1:
                label, conf, occ = "black_pawn", 0.75, 0.8
            elif row == 6:
                label, conf, occ = "white_pawn", 0.75, 0.8
            elif row == 0 and col in (0, 7):
                label, conf, occ = "black_rook", 0.8, 0.85
            elif row == 7 and col in (0, 7):
                label, conf, occ = "white_rook", 0.8, 0.85
            elif row == 0 and col == 4:
                label, conf, occ = "black_king", 0.85, 0.9
            elif row == 7 and col == 4:
                label, conf, occ = "white_king", 0.85, 0.9
            squares.append(_sq(name, row, col, label, conf, occ))
    return squares


class TestHypothesisEngine:
    def test_generates_orientation_variants(self) -> None:
        hyps = generate_board_hypotheses(_starting_finalized(), use_stockfish=False)
        names = {h.name for h in hyps}
        assert "normal" in names
        assert "flipped" in names
        assert hyps[0].total_score >= hyps[-1].total_score

    def test_hypotheses_have_fen(self) -> None:
        hyps = generate_board_hypotheses(_starting_finalized(), use_stockfish=False)
        assert all(h.fen for h in hyps)
        assert all("/" in h.fen for h in hyps)
