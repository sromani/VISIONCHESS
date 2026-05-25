"""Aggregate confidence from square predictions and validation issues."""

from __future__ import annotations

from vision.fen.pieces import PieceLabel
from vision.fen.types import FenIssueDetail, Grid8x8


def compute_confidence(grid: Grid8x8, issues: list[FenIssueDetail], *, is_valid: bool) -> float:
    """Harmonic mean of occupied-square confidence, penalized by issue severity."""
    occupied = [cell.confidence for row in grid for cell in row if cell.label != PieceLabel.EMPTY]

    if not occupied:
        base = 0.0
    else:
        base = len(occupied) / sum(1.0 / max(conf, 1e-6) for conf in occupied)

    penalty = 1.0
    for issue in issues:
        penalty *= max(0.0, 1.0 - issue.severity)

    score = base * penalty
    if not is_valid:
        score *= 0.5
    return round(max(0.0, min(score, 1.0)), 4)
