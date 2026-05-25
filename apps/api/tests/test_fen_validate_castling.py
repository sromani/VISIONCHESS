"""Tests for LC2FEN FEN validation and castling inference."""

from __future__ import annotations

import chess

from vision.lc2fen.fen_validate import infer_castling_rights, validate_placement_fen

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_infer_castling_starting_position() -> None:
    board = chess.Board(f"{START} w - - 0 1")
    assert infer_castling_rights(board) == "KQkq"


def test_infer_castling_no_rooks() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert infer_castling_rights(board) == "-"


def test_validate_placement_includes_castling() -> None:
    result = validate_placement_fen(START)
    assert result.board_ready
    assert result.interactive_fen is not None
    assert "KQkq" in result.interactive_fen
