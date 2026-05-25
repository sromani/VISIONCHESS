"""Unit tests for FEN builder."""

from __future__ import annotations

import chess
import pytest

from vision.fen import FenBuilder, FenIssue, Grid8x8, InvalidGridError, SquarePrediction
from vision.fen.builder import FenBuilderConfig, build_fen_from_labels
from vision.fen.compression import compress_rank
from vision.fen.types import grid_from_labels, normalize_grid


def _starting_labels() -> list[list[str]]:
    return [
        ["black_rook", "black_knight", "black_bishop", "black_queen", "black_king", "black_bishop", "black_knight", "black_rook"],
        ["black_pawn"] * 8,
        ["empty"] * 8,
        ["empty"] * 8,
        ["empty"] * 8,
        ["empty"] * 8,
        ["white_pawn"] * 8,
        ["white_rook", "white_knight", "white_bishop", "white_queen", "white_king", "white_bishop", "white_knight", "white_rook"],
    ]


class TestCompression:
    def test_empty_rank(self) -> None:
        assert compress_rank([None] * 8) == "8"

    def test_mixed_rank(self) -> None:
        symbols: list[str | None] = [None, None, "n", None, None, None, "k", None]
        assert compress_rank(symbols) == "2n3k1"

    def test_full_rank_no_compression(self) -> None:
        assert compress_rank(list("rnbqkbnr")) == "rnbqkbnr"


class TestStartingPosition:
    def test_build_starting_position(self) -> None:
        result = build_fen_from_labels(_starting_labels())
        expected_placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        assert result.placement == expected_placement
        assert result.fen.startswith(expected_placement)
        assert result.is_valid is True
        assert result.confidence > 0.9
        assert chess.Board(result.fen).is_valid()

    def test_active_color(self) -> None:
        builder = FenBuilder(FenBuilderConfig(active_color="b"))
        result = builder.build(grid_from_labels(_starting_labels()))
        assert " b " in result.fen


class TestKingValidation:
    def test_missing_kings_invalid(self) -> None:
        labels = [["empty"] * 8 for _ in range(8)]
        result = build_fen_from_labels(labels)
        assert result.is_valid is False
        assert any(i.code == FenIssue.MISSING_WHITE_KING for i in result.issues)
        assert any(i.code == FenIssue.MISSING_BLACK_KING for i in result.issues)
        assert any(i.code == FenIssue.EMPTY_BOARD for i in result.issues)

    def test_extra_kings_repaired(self) -> None:
        labels = [["empty"] * 8 for _ in range(8)]
        labels[0][4] = "black_king"
        labels[0][5] = "black_king"
        labels[7][4] = "white_king"
        labels[7][5] = "white_king"
        confidences = [[0.9 if c != "empty" else 1.0 for c in row] for row in labels]
        confidences[0][4] = 0.95
        confidences[0][5] = 0.4
        confidences[7][4] = 0.88
        confidences[7][5] = 0.3

        result = build_fen_from_labels(labels, confidences)
        kings = sum(
            1
            for row in result.repaired_grid
            for cell in row
            if cell.label.endswith("_king")
        )
        assert kings == 2
        assert any(i.code == FenIssue.EXTRA_WHITE_KING for i in result.issues)
        assert any(i.code == FenIssue.EXTRA_BLACK_KING for i in result.issues)
        assert any(i.code == FenIssue.KING_REPAIRED for i in result.issues)


class TestPartialErrors:
    def test_unknown_label_becomes_empty(self) -> None:
        labels = [["empty"] * 8 for _ in range(8)]
        labels[3][3] = "unknown_piece"
        labels[0][4] = "black_king"
        labels[7][4] = "white_king"

        result = build_fen_from_labels(labels)
        assert result.repaired_grid[3][3].label == "empty"
        assert any(i.code == FenIssue.UNKNOWN_LABEL for i in result.issues)

    def test_low_confidence_flagged_not_removed(self) -> None:
        labels = [["empty"] * 8 for _ in range(8)]
        labels[0][4] = "black_king"
        labels[7][4] = "white_king"
        labels[4][4] = "white_queen"
        confidences = [[1.0] * 8 for _ in range(8)]
        confidences[4][4] = 0.2

        result = build_fen_from_labels(labels, confidences)
        assert result.repaired_grid[4][4].label == "white_queen"
        assert any(i.code == FenIssue.LOW_SQUARE_CONFIDENCE for i in result.issues)
        assert result.confidence < 1.0

    def test_back_rank_pawn_demoted(self) -> None:
        labels = [["empty"] * 8 for _ in range(8)]
        labels[0][0] = "black_pawn"
        labels[0][4] = "black_king"
        labels[7][4] = "white_king"

        result = build_fen_from_labels(labels)
        assert result.repaired_grid[0][0].label == "empty"
        assert any(i.code == FenIssue.INVALID_PAWN_RANK for i in result.issues)


class TestGridNormalization:
    def test_accepts_string_grid(self) -> None:
        raw = [[("white_king", 0.9) if r == 7 and c == 4 else ("empty", 1.0) for c in range(8)] for r in range(8)]
        raw[0][4] = ("black_king", 0.95)
        grid = normalize_grid(raw)
        result = FenBuilder().build(grid)
        assert "k" in result.placement
        assert "K" in result.placement

    def test_rejects_bad_dimensions(self) -> None:
        with pytest.raises(InvalidGridError):
            normalize_grid([["empty"] * 8] * 7)


class TestConfidence:
    def test_perfect_board_high_confidence(self) -> None:
        result = build_fen_from_labels(_starting_labels())
        assert result.confidence >= 0.95

    def test_damaged_board_lower_confidence(self) -> None:
        good = build_fen_from_labels(_starting_labels())
        labels = _starting_labels()
        labels[0][4] = "black_king"
        labels[0][5] = "black_king"
        bad = build_fen_from_labels(labels)
        assert bad.confidence < good.confidence


class TestStrictMode:
    def test_no_repair_when_partial_disabled(self) -> None:
        labels = [["empty"] * 8 for _ in range(8)]
        labels[0][4] = "black_king"
        labels[0][5] = "black_king"
        labels[7][4] = "white_king"

        builder = FenBuilder(FenBuilderConfig(allow_partial=False))
        result = builder.build(grid_from_labels(labels))
        assert any(i.code == FenIssue.EXTRA_BLACK_KING for i in result.issues)
        assert result.repaired_grid[0][5].label == "black_king"
