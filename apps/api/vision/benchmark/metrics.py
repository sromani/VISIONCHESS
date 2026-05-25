"""Benchmark metrics — per-square, occupancy, confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vision.benchmark.fen_grid import square_name
from vision.fen.pieces import PieceLabel

ALL_LABELS: tuple[str, ...] = (
    PieceLabel.EMPTY,
    PieceLabel.WHITE_PAWN,
    PieceLabel.WHITE_KNIGHT,
    PieceLabel.WHITE_BISHOP,
    PieceLabel.WHITE_ROOK,
    PieceLabel.WHITE_QUEEN,
    PieceLabel.WHITE_KING,
    PieceLabel.BLACK_PAWN,
    PieceLabel.BLACK_KNIGHT,
    PieceLabel.BLACK_BISHOP,
    PieceLabel.BLACK_ROOK,
    PieceLabel.BLACK_QUEEN,
    PieceLabel.BLACK_KING,
)


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {label: {other: 0 for other in ALL_LABELS} for label in ALL_LABELS}


@dataclass(frozen=True, slots=True)
class SquareComparison:
    square: str
    row: int
    col: int
    expected_label: str
    predicted_label: str
    expected_occupied: bool
    predicted_occupied: bool
    label_match: bool
    occupancy_match: bool


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    case_id: str
    image_path: str
    expected_placement: str
    predicted_placement: str
    predicted_fen: str
    fen_is_valid: bool
    pipeline_error: str | None
    localization_ok: bool
    per_square: tuple[SquareComparison, ...]
    piece_accuracy: float
    per_square_accuracy: float
    occupancy_f1: float
    occupancy_precision: float
    occupancy_recall: float
    full_board_exact: bool
    timing_ms: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "image_path": self.image_path,
            "expected_placement": self.expected_placement,
            "predicted_placement": self.predicted_placement,
            "predicted_fen": self.predicted_fen,
            "fen_is_valid": self.fen_is_valid,
            "pipeline_error": self.pipeline_error,
            "localization_ok": self.localization_ok,
            "piece_accuracy": self.piece_accuracy,
            "per_square_accuracy": self.per_square_accuracy,
            "occupancy_f1": self.occupancy_f1,
            "occupancy_precision": self.occupancy_precision,
            "occupancy_recall": self.occupancy_recall,
            "full_board_exact": self.full_board_exact,
            "timing_ms": self.timing_ms,
            "per_square": [
                {
                    "square": s.square,
                    "expected": s.expected_label,
                    "predicted": s.predicted_label,
                    "label_match": s.label_match,
                    "occupancy_match": s.occupancy_match,
                }
                for s in self.per_square
                if not s.label_match or not s.occupancy_match
            ],
        }


@dataclass
class BenchmarkAggregate:
    cases_run: int = 0
    cases_failed: int = 0
    piece_accuracy_sum: float = 0.0
    per_square_accuracy_sum: float = 0.0
    occupancy_f1_sum: float = 0.0
    occupancy_precision_sum: float = 0.0
    occupancy_recall_sum: float = 0.0
    full_board_exact_count: int = 0
    legal_fen_count: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=_empty_confusion)
    case_results: list[BenchmarkCaseResult] = field(default_factory=list)

    @property
    def piece_accuracy(self) -> float:
        ok = self.cases_run - self.cases_failed
        return self.piece_accuracy_sum / ok if ok else 0.0

    @property
    def per_square_accuracy(self) -> float:
        ok = self.cases_run - self.cases_failed
        return self.per_square_accuracy_sum / ok if ok else 0.0

    @property
    def occupancy_f1(self) -> float:
        ok = self.cases_run - self.cases_failed
        return self.occupancy_f1_sum / ok if ok else 0.0

    @property
    def occupancy_precision(self) -> float:
        ok = self.cases_run - self.cases_failed
        return self.occupancy_precision_sum / ok if ok else 0.0

    @property
    def occupancy_recall(self) -> float:
        ok = self.cases_run - self.cases_failed
        return self.occupancy_recall_sum / ok if ok else 0.0

    @property
    def full_board_accuracy(self) -> float:
        ok = self.cases_run - self.cases_failed
        return self.full_board_exact_count / ok if ok else 0.0

    @property
    def legal_fen_rate(self) -> float:
        ok = self.cases_run - self.cases_failed
        return self.legal_fen_count / ok if ok else 0.0

    def add_case(self, result: BenchmarkCaseResult) -> None:
        self.cases_run += 1
        self.case_results.append(result)
        if result.pipeline_error:
            self.cases_failed += 1
            return

        self.piece_accuracy_sum += result.piece_accuracy
        self.per_square_accuracy_sum += result.per_square_accuracy
        self.occupancy_f1_sum += result.occupancy_f1
        self.occupancy_precision_sum += result.occupancy_precision
        self.occupancy_recall_sum += result.occupancy_recall
        if result.full_board_exact:
            self.full_board_exact_count += 1
        if result.fen_is_valid:
            self.legal_fen_count += 1

        for sq in result.per_square:
            exp = sq.expected_label
            pred = sq.predicted_label
            self.confusion[exp][pred] = self.confusion[exp].get(pred, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases_run": self.cases_run,
            "cases_failed": self.cases_failed,
            "piece_accuracy": round(self.piece_accuracy, 4),
            "per_square_accuracy": round(self.per_square_accuracy, 4),
            "occupancy_f1": round(self.occupancy_f1, 4),
            "occupancy_precision": round(self.occupancy_precision, 4),
            "occupancy_recall": round(self.occupancy_recall, 4),
            "full_board_accuracy": round(self.full_board_accuracy, 4),
            "legal_fen_rate": round(self.legal_fen_rate, 4),
            "confusion_matrix": {
                "labels": list(ALL_LABELS),
                "matrix": [[self.confusion[e][p] for p in ALL_LABELS] for e in ALL_LABELS],
            },
            "cases": [c.to_dict() for c in self.case_results],
        }


def compare_grids(
    expected: list[list[str]],
    predicted: list[list[str]],
) -> tuple[tuple[SquareComparison, ...], dict[str, float]]:
    comparisons: list[SquareComparison] = []
    label_matches = 0
    occ_tp = occ_fp = occ_fn = 0
    piece_total = 0
    piece_correct = 0

    for row in range(8):
        for col in range(8):
            exp = expected[row][col]
            pred = predicted[row][col]
            exp_occ = exp != PieceLabel.EMPTY
            pred_occ = pred != PieceLabel.EMPTY

            label_ok = exp == pred
            occ_ok = exp_occ == pred_occ
            if label_ok:
                label_matches += 1

            if exp_occ and pred_occ:
                occ_tp += 1
            elif not exp_occ and pred_occ:
                occ_fp += 1
            elif exp_occ and not pred_occ:
                occ_fn += 1

            if exp_occ:
                piece_total += 1
                if exp == pred:
                    piece_correct += 1

            comparisons.append(
                SquareComparison(
                    square=square_name(row, col),
                    row=row,
                    col=col,
                    expected_label=exp,
                    predicted_label=pred,
                    expected_occupied=exp_occ,
                    predicted_occupied=pred_occ,
                    label_match=label_ok,
                    occupancy_match=occ_ok,
                )
            )

    precision = occ_tp / (occ_tp + occ_fp) if (occ_tp + occ_fp) else 0.0
    recall = occ_tp / (occ_tp + occ_fn) if (occ_tp + occ_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics = {
        "per_square_accuracy": label_matches / 64.0,
        "piece_accuracy": piece_correct / piece_total if piece_total else 0.0,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": f1,
    }
    return tuple(comparisons), metrics
