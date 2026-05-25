"""Types for square classification and FEN assembly."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vision.board.types import BoardGridResult

from vision.fen.types import Grid8x8, SquarePrediction


class Occupancy(StrEnum):
    EMPTY = "empty"
    OCCUPIED = "occupied"


@dataclass(frozen=True, slots=True)
class SquareClassification:
    """Result for one square after empty check + piece classification."""

    row: int
    col: int
    square_name: str
    label: str
    confidence: float
    occupied: bool
    occupancy_score: float
    empty_reason: str | None = None

    def to_prediction(self) -> SquarePrediction:
        if not self.occupied:
            return SquarePrediction(label="empty", confidence=self.occupancy_score)
        return SquarePrediction(label=self.label, confidence=self.confidence)


@dataclass(frozen=True, slots=True)
class OrientationCandidate:
    name: str
    grid: Grid8x8
    fen: str
    legality_score: float
    fen_confidence: float
    is_valid: bool
    active_color: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fen": self.fen,
            "legality_score": self.legality_score,
            "fen_confidence": self.fen_confidence,
            "is_valid": self.is_valid,
            "active_color": self.active_color,
        }


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    fen: str
    placement: str
    confidence: float
    is_valid: bool
    board_ready: bool
    interactive_fen: str | None
    board_matrix: list[list[dict[str, str | float]]]
    orientation: str
    active_color: str
    squares: tuple[SquareClassification, ...]
    grid: Grid8x8
    candidates: tuple[OrientationCandidate, ...]
    classifier_backend: str
    dataset_grid: "BoardGridResult | None" = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "placement": self.placement,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "board_ready": self.board_ready,
            "interactive_fen": self.interactive_fen,
            "board_matrix": self.board_matrix,
            "orientation": self.orientation,
            "active_color": self.active_color,
            "classifier_backend": self.classifier_backend,
            "candidates": [c.to_metadata() for c in self.candidates],
            "squares": [
                {
                    "name": sq.square_name,
                    "label": sq.label if sq.occupied else "empty",
                    "confidence": sq.confidence if sq.occupied else sq.occupancy_score,
                    "occupied": sq.occupied,
                    "occupancy_score": sq.occupancy_score,
                    "empty_reason": sq.empty_reason,
                }
                for sq in self.squares
            ],
        }
