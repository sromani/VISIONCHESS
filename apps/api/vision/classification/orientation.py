"""Board orientation detection and grid flipping."""

from __future__ import annotations

from vision.fen.pieces import PieceLabel
from vision.fen.types import Grid8x8, SquarePrediction


def flip_grid_vertical(grid: Grid8x8) -> Grid8x8:
    """Flip board 180° — rank 8 becomes rank 1."""
    return tuple(tuple(grid[7 - row][col] for col in range(8)) for row in range(8))


def flip_grid_horizontal(grid: Grid8x8) -> Grid8x8:
    """Mirror files a↔h."""
    return tuple(tuple(grid[row][7 - col] for col in range(8)) for row in range(8))


def _piece_color(label: str) -> str | None:
    if label == PieceLabel.EMPTY:
        return None
    if label.startswith("white_"):
        return "white"
    if label.startswith("black_"):
        return "black"
    return None


def _piece_kind(label: str) -> str | None:
    if label == PieceLabel.EMPTY:
        return None
    return label.split("_", 1)[-1]


def orientation_layout_score(grid: Grid8x8) -> float:
    """Higher = white pieces toward bottom (rows 6-7), black toward top (rows 0-1)."""
    white_rows: list[int] = []
    black_rows: list[int] = []
    white_king_row: int | None = None
    black_king_row: int | None = None

    for row in range(8):
        for col in range(8):
            cell = grid[row][col]
            color = _piece_color(cell.label)
            if color == "white":
                white_rows.append(row)
                if _piece_kind(cell.label) == "king":
                    white_king_row = row
            elif color == "black":
                black_rows.append(row)
                if _piece_kind(cell.label) == "king":
                    black_king_row = row

    if not white_rows or not black_rows:
        return 0.0

    white_mean = sum(white_rows) / len(white_rows)
    black_mean = sum(black_rows) / len(black_rows)
    separation = white_mean - black_mean

    score = min(max(separation / 4.0, 0.0), 1.0) * 0.55

    # White king usually on back rank (row 7) or near it
    if white_king_row is not None:
        score += 0.25 * (1.0 - abs(white_king_row - 7) / 7.0)
    if black_king_row is not None:
        score += 0.20 * (1.0 - abs(black_king_row - 0) / 7.0)

    return min(score, 1.0)


def pawn_direction_score(grid: Grid8x8) -> float:
    """White pawns on lower ranks than black pawns is more plausible."""
    white_pawn_rows = [r for r in range(8) for c in range(8) if grid[r][c].label == PieceLabel.WHITE_PAWN]
    black_pawn_rows = [r for r in range(8) for c in range(8) if grid[r][c].label == PieceLabel.BLACK_PAWN]
    if not white_pawn_rows or not black_pawn_rows:
        return 0.5
    return 1.0 if sum(white_pawn_rows) / len(white_pawn_rows) > sum(black_pawn_rows) / len(black_pawn_rows) else 0.0


def infer_active_color(grid: Grid8x8) -> str:
    """Side to move heuristic — more advanced white army → black to move, else white."""
    white_adv = sum(max(0, 6 - r) for r in range(8) for c in range(8) if _piece_color(grid[r][c].label) == "white")
    black_adv = sum(max(0, r - 1) for r in range(8) for c in range(8) if _piece_color(grid[r][c].label) == "black")
    return "b" if white_adv > black_adv + 2 else "w"


def grid_to_labels(grid: Grid8x8) -> list[list[str]]:
    return [[cell.label for cell in row] for row in grid]


def labels_to_grid(labels: list[list[str]], confidences: list[list[float]] | None = None) -> Grid8x8:
    from vision.fen.types import grid_from_labels

    return grid_from_labels(labels, confidences)
