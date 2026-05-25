"""Legacy heuristic backend — dev only, not used in production pipeline."""

from __future__ import annotations

from vision.board.types import BoardGridResult, SquareCrop
from vision.classification.heuristic import HeuristicClassifierConfig, classify_piece_heuristic
from vision.classification.types import SquareClassification


class HeuristicBackend:
    def __init__(self, config: HeuristicClassifierConfig | None = None) -> None:
        self._config = config or HeuristicClassifierConfig()

    @property
    def name(self) -> str:
        return "heuristic"

    def classify_squares(
        self,
        squares: list[SquareCrop],
        *,
        soft: bool = True,
    ) -> list[SquareClassification]:
        results: list[SquareClassification] = []
        for sq in squares:
            label, conf, occupied, occ_score, reason = classify_piece_heuristic(
                sq.image,
                sq.row,
                sq.col,
                self._config,
                skip_occupancy=True,
            )
            results.append(
                SquareClassification(
                    row=sq.row,
                    col=sq.col,
                    square_name=sq.square_name,
                    label=label,
                    confidence=conf,
                    occupied=False if soft else occupied,
                    occupancy_score=occ_score,
                    empty_reason=reason,
                )
            )
        return results

    def classify_grid(self, grid: BoardGridResult) -> list[SquareClassification]:
        return self.classify_squares(list(grid.flat))
