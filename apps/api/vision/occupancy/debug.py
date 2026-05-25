"""Occupancy debug — probability heatmap (soft, pre-finalize)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.types import BoardGridResult
from vision.occupancy.types import OccupancyReport, SquareOccupancyDebug


def render_occupancy_debug(
    grid: BoardGridResult,
    report: OccupancyReport,
    *,
    cell_px: int = 64,
) -> NDArray[np.uint8]:
    """Heatmap of soft occupancy probability — green=low, red=high."""
    board = np.zeros((cell_px * 8, cell_px * 8, 3), dtype=np.uint8)
    debug_by_name = {d.square_name: d for d in report.debug_rows}
    threshold = report.board_threshold

    for sq in grid.flat:
        d = debug_by_name.get(sq.square_name)
        if d is None:
            continue
        y0, x0 = sq.row * cell_px, sq.col * cell_px
        p = d.fused_probability
        t = max(threshold, 0.01)
        ratio = min(p / t, 1.5)
        red = int(min(255, ratio * 180))
        green = int(min(255, (1.0 - min(ratio, 1.0)) * 200))
        board[y0 : y0 + cell_px, x0 : x0 + cell_px] = (40, green, red)
        cv2.putText(
            board,
            f"{p:.2f}",
            (x0 + 3, y0 + cell_px - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return board


def render_occupancy_detail_panel(
    debug_rows: tuple[SquareOccupancyDebug, ...],
    *,
    board_threshold: float = 0.0,
    width: int = 720,
    height: int = 220,
) -> NDArray[np.uint8]:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)
    cv2.putText(
        panel,
        f"SOFT OCCUPANCY (threshold ref={board_threshold:.2f}) — finalized at validation",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    ranked = sorted(debug_rows, key=lambda r: r.fused_probability, reverse=True)
    y = 48
    for d in ranked[:10]:
        line = (
            f"{d.square_name}  fg={d.foreground_score:.2f} sil={d.silhouette_score:.2f} "
            f"p={d.fused_probability:.2f} ml={d.ml_probability}"
        )
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 220, 160), 1, cv2.LINE_AA)
        y += 20
    return panel
