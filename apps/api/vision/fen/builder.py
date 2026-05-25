"""Build a FEN string from an 8×8 detection grid."""

from __future__ import annotations

from dataclasses import dataclass

from vision.fen.compression import assemble_fen, grid_to_placement
from vision.fen.confidence import compute_confidence
from vision.fen.exceptions import InvalidGridError
from vision.fen.pieces import PieceLabel
from vision.fen.repair import repair_kings, repair_pawn_ranks, repair_unknown_labels
from vision.fen.types import (
    FenBuildResult,
    FenIssue,
    FenIssueDetail,
    Grid8x8,
    SquarePrediction,
    normalize_grid,
)
from vision.fen.validation import count_pieces, validate_with_python_chess


@dataclass(frozen=True, slots=True)
class FenBuilderConfig:
    active_color: str = "w"
    castling: str = "-"
    en_passant: str = "-"
    halfmove_clock: int = 0
    fullmove_number: int = 1
    low_confidence_threshold: float = 0.45
    allow_partial: bool = True


class FenBuilder:
    """Convert a detected 8×8 piece grid into a FEN string."""

    def __init__(self, config: FenBuilderConfig | None = None) -> None:
        self._config = config or FenBuilderConfig()

    @property
    def config(self) -> FenBuilderConfig:
        return self._config

    def build(
        self,
        grid: Grid8x8 | list[list[SquarePrediction | tuple[str, float] | str]],
    ) -> FenBuildResult:
        normalized = normalize_grid(grid) if not self._is_grid8x8(grid) else grid
        issues: list[FenIssueDetail] = []

        if count_pieces(normalized) == 0:
            issues.append(
                FenIssueDetail(
                    code=FenIssue.EMPTY_BOARD,
                    message="Board has no detected pieces",
                    severity=0.6,
                )
            )

        working, label_issues = repair_unknown_labels(
            normalized,
            low_confidence_threshold=self._config.low_confidence_threshold,
        )
        issues.extend(label_issues)

        if self._config.allow_partial:
            working, king_issues = repair_kings(working)
            issues.extend(king_issues)
            working, pawn_issues = repair_pawn_ranks(working)
            issues.extend(pawn_issues)
        else:
            from vision.fen.validation import validate_kings, validate_pawn_ranks

            white_kings, black_kings = validate_kings(working)
            if len(white_kings) != 1:
                issues.append(
                    FenIssueDetail(
                        code=FenIssue.MISSING_WHITE_KING if not white_kings else FenIssue.EXTRA_WHITE_KING,
                        message=f"Expected 1 white king, found {len(white_kings)}",
                        squares=tuple(white_kings),
                        severity=0.45,
                    )
                )
            if len(black_kings) != 1:
                issues.append(
                    FenIssueDetail(
                        code=FenIssue.MISSING_BLACK_KING if not black_kings else FenIssue.EXTRA_BLACK_KING,
                        message=f"Expected 1 black king, found {len(black_kings)}",
                        squares=tuple(black_kings),
                        severity=0.45,
                    )
                )
            invalid_pawns = validate_pawn_ranks(working)
            if invalid_pawns:
                issues.append(
                    FenIssueDetail(
                        code=FenIssue.INVALID_PAWN_RANK,
                        message="Pawn detected on back rank",
                        squares=tuple(invalid_pawns),
                        severity=0.2,
                    )
                )

        placement = grid_to_placement(working)
        fen = assemble_fen(
            placement,
            active_color=self._config.active_color,
            castling=self._config.castling,
            en_passant=self._config.en_passant,
            halfmove_clock=self._config.halfmove_clock,
            fullmove_number=self._config.fullmove_number,
        )

        chess_issue = validate_with_python_chess(fen)
        from vision.fen.validation import validate_kings

        white_kings, black_kings = validate_kings(working)
        kings_ok = len(white_kings) == 1 and len(black_kings) == 1
        has_pieces = count_pieces(working) > 0
        is_valid = chess_issue is None and kings_ok and has_pieces
        if chess_issue is not None:
            issues.append(chess_issue)

        confidence = compute_confidence(working, issues, is_valid=is_valid)

        return FenBuildResult(
            fen=fen,
            placement=placement,
            confidence=confidence,
            is_valid=is_valid,
            issues=tuple(issues),
            repaired_grid=working,
        )

    @staticmethod
    def _is_grid8x8(
        grid: Grid8x8 | list[list[SquarePrediction | tuple[str, float] | str]],
    ) -> bool:
        if not isinstance(grid, tuple):
            return False
        if len(grid) != 8:
            return False
        return all(isinstance(row, tuple) and len(row) == 8 for row in grid)


def build_fen_from_labels(
    labels: list[list[str]],
    confidences: list[list[float]] | None = None,
    *,
    active_color: str = "w",
    allow_partial: bool = True,
) -> FenBuildResult:
    """Convenience wrapper for label-only grids."""
    from vision.fen.types import grid_from_labels

    config = FenBuilderConfig(active_color=active_color, allow_partial=allow_partial)
    return FenBuilder(config).build(grid_from_labels(labels, confidences))
