"""Shared soft-probability types for occupancy + classification."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SoftSquarePrediction:
    """Per-square soft state before global board commit."""

    square_name: str
    row: int
    col: int
    occupancy_prob: float
    piece_label: str
    piece_prob: float

    @property
    def joint_prob(self) -> float:
        if self.piece_label == "empty":
            return (1.0 - self.occupancy_prob) * self.piece_prob
        return self.occupancy_prob * self.piece_prob


@dataclass(frozen=True, slots=True)
class BoardHypothesis:
    """One complete board interpretation candidate."""

    name: str
    fen: str
    orientation: str
    legality_score: float
    stockfish_bonus: float
    fen_confidence: float
    is_valid: bool
    active_color: str
    square_labels: dict[str, str] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        return self.legality_score + self.stockfish_bonus
