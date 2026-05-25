"""FEN placement ↔ 8×8 label grid (row 0 = rank 8, col 0 = file a)."""

from __future__ import annotations

from vision.fen.pieces import FEN_TO_LABEL, PieceLabel


def expand_rank(rank: str) -> list[str | None]:
    symbols: list[str | None] = []
    idx = 0
    while idx < len(rank):
        ch = rank[idx]
        if ch.isdigit():
            symbols.extend([None] * int(ch))
            idx += 1
            continue
        symbols.append(ch)
        idx += 1
    if len(symbols) != 8:
        msg = f"rank must expand to 8 squares, got {len(symbols)} from {rank!r}"
        raise ValueError(msg)
    return symbols


def placement_to_labels(placement: str) -> list[list[str]]:
    """Convert FEN piece-placement field to VisionChess labels."""
    ranks = placement.strip().split("/")
    if len(ranks) != 8:
        msg = f"placement must have 8 ranks, got {len(ranks)}"
        raise ValueError(msg)

    grid: list[list[str]] = []
    for rank in ranks:
        row: list[str] = []
        for symbol in expand_rank(rank):
            if symbol is None:
                row.append(PieceLabel.EMPTY)
            else:
                label = FEN_TO_LABEL.get(symbol)
                if label is None:
                    msg = f"unknown FEN symbol: {symbol!r}"
                    raise ValueError(msg)
                row.append(label)
        grid.append(row)
    return grid


def labels_to_placement(grid: list[list[str]]) -> str:
    from vision.fen.compression import compress_rank
    from vision.fen.pieces import to_fen_symbol

    ranks: list[str] = []
    for row in grid:
        symbols = [to_fen_symbol(cell) for cell in row]
        ranks.append(compress_rank(symbols))
    return "/".join(ranks)


def flip_grid_vertical(grid: list[list[str]]) -> list[list[str]]:
    return [list(reversed(row)) for row in reversed(grid)]


def square_name(row: int, col: int) -> str:
    return f"{chr(ord('a') + col)}{8 - row}"
