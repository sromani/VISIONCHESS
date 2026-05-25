"""Validate LC2FEN placement FEN for interactive board use."""

from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class FenValidation:
    fen: str
    interactive_fen: str | None
    is_valid: bool
    board_ready: bool
    confidence: float
    kings_ok: bool
    piece_count: int


def infer_castling_rights(board: chess.Board) -> str:
    """Infer castling availability from king/rook home squares (chess rules)."""
    rights: list[str] = []

    if (
        board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE)
        and board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE)
    ):
        rights.append("K")
    if (
        board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE)
        and board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE)
    ):
        rights.append("Q")
    if (
        board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK)
        and board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK)
    ):
        rights.append("k")
    if (
        board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK)
        and board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK)
    ):
        rights.append("q")

    return "".join(rights) if rights else "-"


def build_interactive_fen(placement: str, *, active_color: str = "w") -> str:
    """Build a full legal FEN with inferred castling rights."""
    board = chess.Board(f"{placement} {active_color} - - 0 1")
    board.set_castling_fen(infer_castling_rights(board))
    return board.fen()


def validate_placement_fen(fen: str, *, min_pieces: int = 4, min_confidence: float = 0.5) -> FenValidation:
    """Check whether a piece-placement FEN is usable on the interactive board."""
    piece_count = sum(1 for ch in fen if ch.isalpha())
    confidence = min(1.0, piece_count / 32.0)

    try:
        board = chess.Board(f"{fen} w - - 0 1")
        board.set_castling_fen(infer_castling_rights(board))
    except ValueError:
        return FenValidation(
            fen=fen,
            interactive_fen=None,
            is_valid=False,
            board_ready=False,
            confidence=confidence,
            kings_ok=False,
            piece_count=piece_count,
        )

    kings = board.pieces(chess.KING, chess.WHITE) | board.pieces(chess.KING, chess.BLACK)
    kings_ok = len(kings) == 2
    legal = board.is_valid()
    board_ready = (
        legal
        and kings_ok
        and piece_count >= min_pieces
        and confidence >= min_confidence
    )

    return FenValidation(
        fen=fen,
        interactive_fen=board.fen() if board_ready else None,
        is_valid=legal,
        board_ready=board_ready,
        confidence=confidence,
        kings_ok=kings_ok,
        piece_count=piece_count,
    )
