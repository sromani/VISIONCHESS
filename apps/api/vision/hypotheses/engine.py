"""Generate and rank board hypotheses from finalized square classifications."""

from __future__ import annotations

from dataclasses import dataclass

from vision.classification.legality import score_grid
from vision.classification.orientation import flip_grid_vertical, infer_active_color
from vision.classification.pipeline import _squares_to_grid
from vision.classification.types import SquareClassification
from vision.core.types import BoardHypothesis
from vision.fen.types import Grid8x8, SquarePrediction
from vision.validation.scorer import score_hypothesis_with_stockfish


@dataclass(frozen=True, slots=True)
class HypothesisEngine:
    """Chesscog-style: finalized squares → orientation variants → python-chess + Stockfish."""

    stockfish_path: str | None = None
    use_stockfish: bool = True
    allow_partial_fen: bool = True

    def generate(self, finalized: list[SquareClassification]) -> list[BoardHypothesis]:
        return generate_board_hypotheses(
            finalized,
            stockfish_path=self.stockfish_path,
            use_stockfish=self.use_stockfish,
            allow_partial=self.allow_partial_fen,
        )


def generate_board_hypotheses(
    finalized: list[SquareClassification],
    *,
    stockfish_path: str | None = None,
    use_stockfish: bool = True,
    allow_partial: bool = True,
) -> list[BoardHypothesis]:
    """Score orientation + pruned variants with python-chess and Stockfish."""
    pred_grid = _squares_to_grid(finalized)
    hypotheses: list[BoardHypothesis] = []

    for orient_name, oriented in (
        ("normal", pred_grid),
        ("flipped", flip_grid_vertical(pred_grid)),
    ):
        hypotheses.append(
            _score_orientation(oriented, orient_name, stockfish_path, use_stockfish, allow_partial)
        )

    pruned = _pruned_low_confidence_hypothesis(pred_grid, allow_partial=allow_partial)
    if pruned is not None:
        sf_bonus = 0.0
        if use_stockfish and stockfish_path and pruned.is_valid:
            sf_bonus = score_hypothesis_with_stockfish(pruned.fen, stockfish_path)
        hypotheses.append(
            BoardHypothesis(
                name=pruned.name,
                fen=pruned.fen,
                orientation=pruned.orientation,
                legality_score=pruned.legality_score,
                stockfish_bonus=sf_bonus,
                fen_confidence=pruned.fen_confidence,
                is_valid=pruned.is_valid,
                active_color=pruned.active_color,
                square_labels=pruned.square_labels,
            )
        )

    hypotheses.sort(key=lambda h: h.total_score, reverse=True)
    return hypotheses


def _score_orientation(
    grid: Grid8x8,
    name: str,
    stockfish_path: str | None,
    use_stockfish: bool,
    allow_partial: bool,
) -> BoardHypothesis:
    active = infer_active_color(grid)
    legality = score_grid(grid, active_color=active, allow_partial=allow_partial)
    sf_bonus = 0.0
    if use_stockfish and stockfish_path and legality.fen_result.is_valid:
        sf_bonus = score_hypothesis_with_stockfish(legality.fen_result.fen, stockfish_path)
    return BoardHypothesis(
        name=name,
        fen=legality.fen_result.fen,
        orientation=name,
        legality_score=legality.total,
        stockfish_bonus=sf_bonus,
        fen_confidence=legality.fen_result.confidence,
        is_valid=legality.is_valid,
        active_color=active,
        square_labels=_grid_to_square_labels(legality.fen_result.repaired_grid),
    )


def _grid_to_square_labels(grid: Grid8x8) -> dict[str, str]:
    files = "abcdefgh"
    labels: dict[str, str] = {}
    for row in range(8):
        for col in range(8):
            cell = grid[row][col]
            name = f"{files[col]}{8 - row}"
            labels[name] = cell.label
    return labels


def _pruned_low_confidence_hypothesis(
    grid: Grid8x8,
    *,
    allow_partial: bool,
) -> BoardHypothesis | None:
    adjusted: list[list[SquarePrediction]] = []
    changed = False
    for row in grid:
        new_row: list[SquarePrediction] = []
        for cell in row:
            if cell.label != "empty" and cell.confidence < 0.42:
                new_row.append(SquarePrediction("empty", cell.confidence))
                changed = True
            else:
                new_row.append(cell)
        adjusted.append(new_row)

    if not changed:
        return None

    adj_grid = tuple(tuple(r) for r in adjusted)
    hyp = _score_orientation(adj_grid, "pruned_low_confidence", None, False, allow_partial)
    return BoardHypothesis(
        name=hyp.name,
        fen=hyp.fen,
        orientation=hyp.orientation,
        legality_score=hyp.legality_score - 5.0,
        stockfish_bonus=hyp.stockfish_bonus,
        fen_confidence=hyp.fen_confidence,
        is_valid=hyp.is_valid,
        active_color=hyp.active_color,
        square_labels=hyp.square_labels,
    )
