"""Repair partial detection errors before FEN assembly."""

from __future__ import annotations

from dataclasses import replace

from vision.fen.pieces import PieceLabel, is_known_label
from vision.fen.types import FenIssue, FenIssueDetail, Grid8x8, SquarePrediction


def repair_unknown_labels(
    grid: Grid8x8,
    *,
    low_confidence_threshold: float,
) -> tuple[Grid8x8, list[FenIssueDetail]]:
    issues: list[FenIssueDetail] = []
    repaired_rows: list[tuple[SquarePrediction, ...]] = []

    for row_idx, row in enumerate(grid):
        repaired_cells: list[SquarePrediction] = []
        for col_idx, cell in enumerate(row):
            updated = cell
            if not is_known_label(cell.label):
                issues.append(
                    FenIssueDetail(
                        code=FenIssue.UNKNOWN_LABEL,
                        message=f"Unknown label '{cell.label}' treated as empty",
                        squares=((row_idx, col_idx),),
                        severity=0.15,
                    )
                )
                updated = SquarePrediction(label=PieceLabel.EMPTY, confidence=cell.confidence)
            elif cell.label != PieceLabel.EMPTY and cell.confidence < low_confidence_threshold:
                issues.append(
                    FenIssueDetail(
                        code=FenIssue.LOW_SQUARE_CONFIDENCE,
                        message=(
                            f"Low confidence ({cell.confidence:.2f}) on "
                            f"{cell.label} at ({row_idx}, {col_idx})"
                        ),
                        squares=((row_idx, col_idx),),
                        severity=0.08,
                    )
                )
            repaired_cells.append(updated)
        repaired_rows.append(tuple(repaired_cells))

    return tuple(repaired_rows), issues


def repair_kings(grid: Grid8x8) -> tuple[Grid8x8, list[FenIssueDetail]]:
    """Keep the highest-confidence king per color; demote extras to empty."""
    issues: list[FenIssueDetail] = []
    mutable = [[replace(cell) for cell in row] for row in grid]

    for color, label in (("white", PieceLabel.WHITE_KING), ("black", PieceLabel.BLACK_KING)):
        king_cells: list[tuple[int, int, float]] = []
        for row_idx, row in enumerate(mutable):
            for col_idx, cell in enumerate(row):
                if cell.label == label:
                    king_cells.append((row_idx, col_idx, cell.confidence))

        if not king_cells:
            code = FenIssue.MISSING_WHITE_KING if color == "white" else FenIssue.MISSING_BLACK_KING
            issues.append(
                FenIssueDetail(
                    code=code,
                    message=f"Missing {color} king",
                    severity=0.45,
                )
            )
            continue

        if len(king_cells) == 1:
            continue

        code = FenIssue.EXTRA_WHITE_KING if color == "white" else FenIssue.EXTRA_BLACK_KING
        king_cells.sort(key=lambda item: item[2], reverse=True)
        keep = king_cells[0]
        demoted: list[tuple[int, int]] = []

        for row_idx, col_idx, _ in king_cells[1:]:
            mutable[row_idx][col_idx] = SquarePrediction(
                label=PieceLabel.EMPTY,
                confidence=mutable[row_idx][col_idx].confidence,
            )
            demoted.append((row_idx, col_idx))

        issues.append(
            FenIssueDetail(
                code=code,
                message=f"Multiple {color} kings detected",
                squares=tuple((row, col) for row, col, _ in king_cells),
                severity=0.25,
            )
        )
        issues.append(
            FenIssueDetail(
                code=FenIssue.KING_REPAIRED,
                message=f"Kept {color} king at ({keep[0]}, {keep[1]})",
                squares=((keep[0], keep[1]),),
                severity=0.05,
            )
        )
        if demoted:
            issues[-2] = FenIssueDetail(
                code=issues[-2].code,
                message=issues[-2].message + f"; demoted {len(demoted)} square(s)",
                squares=issues[-2].squares,
                severity=issues[-2].severity,
            )

    repaired = tuple(tuple(row) for row in mutable)
    return repaired, issues


def repair_pawn_ranks(grid: Grid8x8) -> tuple[Grid8x8, list[FenIssueDetail]]:
    """Demote back-rank pawns — common misclassification near edges."""
    from vision.fen.validation import validate_pawn_ranks

    invalid = validate_pawn_ranks(grid)
    if not invalid:
        return grid, []

    mutable = [[replace(cell) for cell in row] for row in grid]
    for row_idx, col_idx in invalid:
        cell = mutable[row_idx][col_idx]
        mutable[row_idx][col_idx] = SquarePrediction(label=PieceLabel.EMPTY, confidence=cell.confidence)

    return tuple(tuple(row) for row in mutable), [
        FenIssueDetail(
            code=FenIssue.INVALID_PAWN_RANK,
            message="Pawn on back rank demoted to empty",
            squares=tuple(invalid),
            severity=0.12,
        )
    ]
