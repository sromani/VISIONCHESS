"""Global board validation and Stockfish hypothesis scoring."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from vision.classification.legality import LegalityScore, score_grid, quick_stockfish_plausibility
from vision.fen.types import Grid8x8


@dataclass(frozen=True, slots=True)
class BoardValidator:
    """python-chess validation wrapper for production pipeline."""

    allow_partial_fen: bool = True

    def score(self, grid: Grid8x8, *, active_color: str = "w", layout_score: float = 0.0) -> LegalityScore:
        return score_grid(
            grid,
            active_color=active_color,
            layout_score=layout_score,
            allow_partial=self.allow_partial_fen,
        )

    def is_legal_fen(self, fen: str) -> bool:
        try:
            board = chess.Board(fen)
            return board.is_valid()
        except ValueError:
            return False

    def piece_count_sane(self, fen: str) -> bool:
        try:
            board = chess.Board(fen)
        except ValueError:
            return False
        if len(board.piece_map()) > 32:
            return False
        kings = sum(1 for p in board.piece_map().values() if p.piece_type == chess.KING)
        return kings == 2


def score_hypothesis_with_stockfish(fen: str, stockfish_path: str, *, depth: int = 10) -> float:
    """Stockfish plausibility bonus for hypothesis ranking."""
    if not stockfish_path:
        return 0.0
    try:
        import chess.engine

        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(fen)
            if not board.is_valid():
                return -10.0
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
            score = info.get("score")
            if score is None:
                return 0.0
            white = score.white()
            if white.is_mate() and abs(white.mate()) <= 1:
                return -5.0
            cp = white.score(mate_score=10000)
            if cp is None:
                return 2.0
            if abs(cp) > 8000:
                return -3.0
            return 2.0 + min(abs(cp) / 4000.0, 1.0)
    except Exception:
        return quick_stockfish_plausibility(fen, stockfish_path)
