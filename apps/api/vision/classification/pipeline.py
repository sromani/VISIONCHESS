"""End-to-end: warped board → classify → orient → FEN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision.board.grid import BoardGridExtractor
from vision.board.grid_config import GridExtractorConfig
from vision.classification.backend_protocol import PieceClassifierBackend
from vision.classification.classifier import ClassifierConfig, create_classifier, resolve_model_path
from vision.classification.legality import LegalityScore, quick_stockfish_plausibility, score_grid
from vision.classification.orientation import (
    flip_grid_vertical,
    infer_active_color,
    orientation_layout_score,
    pawn_direction_score,
)
from vision.classification.types import ClassificationResult, OrientationCandidate, SquareClassification
from vision.fen.types import Grid8x8, SquarePrediction
from vision.classification.square_quality import DatasetSquareConfig, normalize_grid_for_dataset
from vision.fen.validation import count_pieces


@dataclass(frozen=True, slots=True)
class ClassificationPipelineConfig:
    margin_ratio: float = 0.10
    model_path: str | None = None
    stockfish_path: str | None = None
    use_stockfish_tiebreak: bool = False
    allow_partial_fen: bool = True
    min_fen_confidence: float = 0.35
    min_pieces_for_board: int = 2
    dataset_square: DatasetSquareConfig = DatasetSquareConfig()


class ClassificationPipeline:
    """Split warped board, classify pieces, resolve orientation, build FEN."""

    def __init__(
        self,
        config: ClassificationPipelineConfig | None = None,
        classifier: PieceClassifierBackend | None = None,
    ) -> None:
        self._config = config or ClassificationPipelineConfig()
        grid_cfg = GridExtractorConfig(margin_ratio=self._config.margin_ratio)
        self._grid_extractor = BoardGridExtractor(grid_cfg)
        self._classifier = classifier or self._build_classifier()

    @property
    def classifier_backend(self) -> str:
        return self._classifier.name

    def run(self, warped_board: NDArray[np.uint8], board_size: int | None = None) -> ClassificationResult:
        grid_result = self._grid_extractor.extract(warped_board, board_size, uniform=True)
        dataset_grid = normalize_grid_for_dataset(grid_result, self._config.dataset_square)
        square_results = self._classifier.classify_grid(dataset_grid)
        pred_grid = _squares_to_grid(square_results)

        candidates = self._evaluate_orientations(pred_grid)
        best = max(candidates, key=lambda c: c.legality_score)

        legality = score_grid(
            best.grid,
            active_color=best.active_color,
            allow_partial=self._config.allow_partial_fen,
        )

        board_ready = _is_board_ready(legality, self._config)
        board_matrix = grid_to_matrix(legality.fen_result.repaired_grid)

        return ClassificationResult(
            fen=legality.fen_result.fen,
            placement=legality.fen_result.placement,
            confidence=legality.fen_result.confidence,
            is_valid=legality.is_valid,
            board_ready=board_ready,
            interactive_fen=legality.fen_result.fen if board_ready else None,
            board_matrix=board_matrix,
            orientation=best.name,
            active_color=best.active_color,
            squares=tuple(square_results),
            grid=legality.fen_result.repaired_grid,
            candidates=tuple(candidates),
            classifier_backend=self._classifier.name,
            dataset_grid=dataset_grid,
        )

    def _build_classifier(self) -> PieceClassifierBackend:
        model = resolve_model_path(self._config.model_path)
        return create_classifier(
            ClassifierConfig(
                backend="auto",
                model_path=model,
                allow_heuristic_fallback=True,
            ),
        )

    def _evaluate_orientations(self, grid: Grid8x8) -> list[OrientationCandidate]:
        variants: list[tuple[str, Grid8x8]] = [
            ("normal", grid),
            ("flipped", flip_grid_vertical(grid)),
        ]

        candidates: list[OrientationCandidate] = []
        for name, oriented in variants:
            active = infer_active_color(oriented)
            layout = 0.6 * orientation_layout_score(oriented) + 0.4 * pawn_direction_score(oriented)
            legality = score_grid(
                oriented,
                active_color=active,
                layout_score=layout,
                allow_partial=self._config.allow_partial_fen,
            )
            total = legality.total
            if self._config.use_stockfish_tiebreak:
                total += quick_stockfish_plausibility(
                    legality.fen_result.fen,
                    self._config.stockfish_path,
                )

            candidates.append(
                OrientationCandidate(
                    name=name,
                    grid=legality.fen_result.repaired_grid,
                    fen=legality.fen_result.fen,
                    legality_score=total,
                    fen_confidence=legality.fen_result.confidence,
                    is_valid=legality.is_valid,
                    active_color=active,
                )
            )

        return candidates


def _squares_to_grid(squares: list[SquareClassification]) -> Grid8x8:
    cells: list[list[SquarePrediction]] = [[SquarePrediction("empty", 0.0) for _ in range(8)] for _ in range(8)]
    for sq in squares:
        cells[sq.row][sq.col] = sq.to_prediction()
    return tuple(tuple(row) for row in cells)


def grid_to_matrix(grid: Grid8x8) -> list[list[dict[str, str | float]]]:
    """Validated 8×8 matrix for API — row 0 = rank 8."""
    matrix: list[list[dict[str, str | float]]] = []
    for row in grid:
        matrix.append(
            [{"label": cell.label, "confidence": round(cell.confidence, 4)} for cell in row]
        )
    return matrix


def _is_board_ready(legality: LegalityScore, config: ClassificationPipelineConfig) -> bool:
    if not legality.is_valid:
        return False
    if not legality.kings_ok or not legality.piece_counts_ok:
        return False
    if legality.fen_result.confidence < config.min_fen_confidence:
        return False
    if count_pieces(legality.fen_result.repaired_grid) < config.min_pieces_for_board:
        return False
    return True
