"""Compress ranks and build the piece-placement field."""

from __future__ import annotations

from vision.fen.pieces import to_fen_symbol
from vision.fen.types import Grid8x8


def compress_rank(symbols: list[str | None]) -> str:
    """Convert eight square symbols into one FEN rank with empty-run compression."""
    if len(symbols) != 8:
        msg = f"rank must have 8 squares, got {len(symbols)}"
        raise ValueError(msg)

    parts: list[str] = []
    empty_run = 0

    for symbol in symbols:
        if symbol is None:
            empty_run += 1
            continue
        if empty_run:
            parts.append(str(empty_run))
            empty_run = 0
        parts.append(symbol)

    if empty_run:
        parts.append(str(empty_run))

    return "".join(parts) if parts else "8"


def grid_to_placement(grid: Grid8x8) -> str:
    """Build the piece-placement component (rows 8 → 1)."""
    ranks: list[str] = []
    for row in grid:
        symbols = [to_fen_symbol(cell.label) for cell in row]
        ranks.append(compress_rank(symbols))
    return "/".join(ranks)


def assemble_fen(
    placement: str,
    *,
    active_color: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove_clock: int = 0,
    fullmove_number: int = 1,
) -> str:
    return f"{placement} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
