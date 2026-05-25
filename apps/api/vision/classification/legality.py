"""Chess legality scoring for orientation selection."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from vision.fen.builder import FenBuilder, FenBuilderConfig
from vision.fen.pieces import KING_LABELS, PAWN_LABELS, PieceLabel
from vision.fen.types import FenBuildResult, Grid8x8
from vision.fen.validation import count_pieces, validate_kings, validate_pawn_ranks


@dataclass(frozen=True, slots=True)
class LegalityScore:
    total: float
    is_valid: bool
    fen_result: FenBuildResult
    kings_ok: bool
    pawns_ok: bool
    piece_counts_ok: bool
    layout_score: float

    def to_metadata(self) -> dict:
        return {
            "total": self.total,
            "is_valid": self.is_valid,
            "kings_ok": self.kings_ok,
            "pawns_ok": self.pawns_ok,
            "piece_counts_ok": self.piece_counts_ok,
            "layout_score": self.layout_score,
        }


MAX_PER_COLOR = {
    "pawn": 8,
    "knight": 2,
    "bishop": 2,
    "rook": 2,
    "queen": 1,
    "king": 1,
}


def count_by_kind(grid: Grid8x8, color: str) -> dict[str, int]:
    counts = {k: 0 for k in MAX_PER_COLOR}
    for row in grid:
        for cell in row:
            if cell.label == PieceLabel.EMPTY:
                continue
            prefix = f"{color}_"
            if cell.label.startswith(prefix):
                kind = cell.label.removeprefix(prefix)
                if kind in counts:
                    counts[kind] += 1
    return counts


def piece_counts_plausible(grid: Grid8x8) -> bool:
    for color in ("white", "black"):
        counts = count_by_kind(grid, color)
        for kind, maximum in MAX_PER_COLOR.items():
            if counts[kind] > maximum:
                return False
    return True


def score_grid(
    grid: Grid8x8,
    *,
    active_color: str = "w",
    layout_score: float = 0.0,
    allow_partial: bool = True,
) -> LegalityScore:
    from vision.classification.orientation import orientation_layout_score, pawn_direction_score

    layout = layout_score if layout_score > 0 else (
        0.6 * orientation_layout_score(grid) + 0.4 * pawn_direction_score(grid)
    )

    white_kings, black_kings = validate_kings(grid)
    kings_ok = len(white_kings) == 1 and len(black_kings) == 1
    pawns_ok = len(validate_pawn_ranks(grid)) == 0
    counts_ok = piece_counts_plausible(grid)

    builder = FenBuilder(FenBuilderConfig(active_color=active_color, allow_partial=allow_partial))
    fen_result = builder.build(grid)

    total = 0.0
    if fen_result.is_valid:
        total += 50.0
    if kings_ok:
        total += 20.0
    if pawns_ok:
        total += 10.0
    if counts_ok:
        total += 10.0
    total += layout * 15.0
    total += fen_result.confidence * 10.0
    total -= len(fen_result.issues) * 3.0

    # python-chess position quality
    try:
        board = chess.Board(fen_result.fen)
        if board.is_valid():
            total += 5.0
        king_count = sum(1 for p in board.piece_map().values() if p.piece_type == chess.KING)
        if king_count == 2 and not board.is_check():
            total += 3.0
    except ValueError:
        total -= 15.0

    if count_pieces(grid) == 0:
        total = 0.0

    return LegalityScore(
        total=total,
        is_valid=fen_result.is_valid and kings_ok,
        fen_result=fen_result,
        kings_ok=kings_ok,
        pawns_ok=pawns_ok,
        piece_counts_ok=counts_ok,
        layout_score=layout,
    )


def quick_stockfish_plausibility(fen: str, stockfish_path: str | None = None) -> float:
    """Optional Stockfish sanity check — returns bonus score or 0 if unavailable."""
    if not stockfish_path:
        return 0.0
    try:
        import chess.engine

        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=8), multipv=1)
            score = info.get("score")
            if score is None:
                return 0.0
            white = score.white()
            if white.is_mate() and abs(white.mate()) <= 1:
                return -5.0
            return 2.0
    except Exception:
        return 0.0
