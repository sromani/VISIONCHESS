"""Domain types for FEN construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from vision.fen.exceptions import InvalidGridError


class FenIssue(StrEnum):
    UNKNOWN_LABEL = "unknown_label"
    LOW_SQUARE_CONFIDENCE = "low_square_confidence"
    MISSING_WHITE_KING = "missing_white_king"
    MISSING_BLACK_KING = "missing_black_king"
    EXTRA_WHITE_KING = "extra_white_king"
    EXTRA_BLACK_KING = "extra_black_king"
    KING_REPAIRED = "king_repaired"
    INVALID_PAWN_RANK = "invalid_pawn_rank"
    ILLEGAL_POSITION = "illegal_position"
    EMPTY_BOARD = "empty_board"


@dataclass(frozen=True, slots=True)
class SquarePrediction:
    label: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence must be in [0, 1], got {self.confidence}"
            raise ValueError(msg)


Grid8x8 = tuple[tuple[SquarePrediction, ...], ...]


@dataclass(frozen=True, slots=True)
class FenIssueDetail:
    code: FenIssue
    message: str
    squares: tuple[tuple[int, int], ...] = ()
    severity: float = 0.0  # confidence penalty in [0, 1]


@dataclass(frozen=True, slots=True)
class FenBuildResult:
    fen: str
    placement: str
    confidence: float
    is_valid: bool
    issues: tuple[FenIssueDetail, ...]
    repaired_grid: Grid8x8

    def to_metadata(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "placement": self.placement,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "issues": [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "squares": [list(sq) for sq in issue.squares],
                    "severity": issue.severity,
                }
                for issue in self.issues
            ],
        }


def normalize_grid(raw: list[list[SquarePrediction | tuple[str, float] | str]]) -> Grid8x8:
    """Coerce common input shapes into a fixed 8×8 grid."""
    if len(raw) != 8 or any(len(row) != 8 for row in raw):
        msg = f"expected 8×8 grid, got {len(raw)} rows"
        raise InvalidGridError(msg)

    normalized: list[tuple[SquarePrediction, ...]] = []
    for row in raw:
        cells: list[SquarePrediction] = []
        for cell in row:
            cells.append(_coerce_square(cell))
        normalized.append(tuple(cells))
    return tuple(normalized)


def _coerce_square(cell: SquarePrediction | tuple[str, float] | str) -> SquarePrediction:
    if isinstance(cell, SquarePrediction):
        return cell
    if isinstance(cell, tuple):
        label, confidence = cell
        return SquarePrediction(label=label, confidence=float(confidence))
    return SquarePrediction(label=cell)


def empty_grid(label: str = "empty", confidence: float = 1.0) -> Grid8x8:
    cell = SquarePrediction(label=label, confidence=confidence)
    return tuple(tuple(cell for _ in range(8)) for _ in range(8))


def grid_from_labels(labels: list[list[str]], confidences: list[list[float]] | None = None) -> Grid8x8:
    if confidences is None:
        confidences = [[1.0] * 8 for _ in range(8)]
    raw: list[list[tuple[str, float]]] = [
        [(labels[r][c], confidences[r][c]) for c in range(8)] for r in range(8)
    ]
    return normalize_grid(raw)
