"""Tests for final occupancy binarization at validation."""

from __future__ import annotations

from vision.classification.types import SquareClassification
from vision.occupancy.finalize import finalize_board
from vision.occupancy.config import OccupancyConfig
from vision.occupancy.types import OccupancyResult


def _occ(name: str, prob: float) -> OccupancyResult:
    return OccupancyResult(
        occupied=False,
        score=prob,
        probability=prob,
        foreground_score=prob,
        silhouette_score=prob,
        edge_score=0.0,
        entropy_score=0.0,
        center_activation=0.0,
        reason="soft_pending",
    )


def _sq(name: str, row: int, col: int, label: str, conf: float) -> SquareClassification:
    return SquareClassification(
        row=row,
        col=col,
        square_name=name,
        label=label,
        confidence=conf,
        occupied=False,
        occupancy_score=0.5,
    )


class TestFinalizeBoard:
    def test_empty_board_mostly_empty(self) -> None:
        occ = {f"s{i}": _occ(f"s{i}", 0.05 + i * 0.001) for i in range(64)}
        squares = [_sq(f"s{i}", i // 8, i % 8, "empty", 0.9) for i in range(64)]
        finalized, _, _ = finalize_board(squares, occ, OccupancyConfig())
        assert sum(1 for s in finalized if s.occupied) <= 12

    def test_piece_cluster_becomes_occupied(self) -> None:
        occ: dict[str, OccupancyResult] = {}
        squares: list[SquareClassification] = []
        for i in range(64):
            name = f"s{i}"
            prob = 0.55 if i >= 48 else 0.08
            occ[name] = _occ(name, prob)
            squares.append(_sq(name, i // 8, i % 8, "white_pawn", 0.75))

        finalized, threshold, _ = finalize_board(squares, occ, OccupancyConfig())
        occupied = [s for s in finalized if s.occupied]
        assert len(occupied) >= 8
        assert threshold < 0.55
        assert all(s.label != "empty" for s in occupied)

    def test_sets_occupied_only_after_finalize(self) -> None:
        occ = {"a1": _occ("a1", 0.7)}
        squares = [_sq("a1", 0, 0, "white_rook", 0.8)]
        finalized, _, _ = finalize_board(squares, occ, OccupancyConfig())
        assert finalized[0].occupied
        assert squares[0].occupied is False
