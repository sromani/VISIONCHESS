"""Phase 3 — upscale rectified board, then extract high-res square crops."""

from __future__ import annotations

import cv2
import numpy as np

from vision.board.grid import BoardGridExtractor
from vision.board.grid_solver import compose_board_preview
from vision.board.types import BoardGridResult
from vision.board.upscale import upscale_rectified_board
from vision.scanner.context import ScanContext


def run_extraction(ctx: ScanContext) -> BoardGridResult:
    """Compose rectified board → super-resolution upscale → uniform 8×8 split."""
    if ctx.grid_solve is None:
        msg = "mesh rectification must run before extraction"
        raise ValueError(msg)

    cfg = ctx.config.grid
    rectified = compose_board_preview(ctx.original_bgr, ctx.grid_solve)

    upscale_size = cfg.upscale_size if cfg.upscale_enabled else rectified.shape[0]
    if cfg.upscale_enabled and upscale_size > rectified.shape[0]:
        rectified = upscale_rectified_board(rectified, upscale_size)

    ctx.rectified_board = rectified

    extractor = BoardGridExtractor(cfg)
    raw = extractor.extract_uniform(rectified, board_size=rectified.shape[0])
    ctx.raw_grid = raw

    ctx.metadata["extraction"] = {
        "method": "rectified_upscale_uniform",
        "rectified_size": int(rectified.shape[0]),
        "upscale_enabled": cfg.upscale_enabled,
        "upscale_size": upscale_size,
        "raw_square_px": raw.square_size,
        "next_stage": "crop_quality",
    }

    if ctx.config.collect_debug:
        ctx.add_debug("rectified_upscaled", rectified)
        ctx.add_debug("square_extraction", _montage(raw, cell_px=min(128, raw.square_size)))

    return raw


def _montage(grid: BoardGridResult, cell_px: int = 64) -> np.ndarray:
    board = np.zeros((cell_px * 8, cell_px * 8, 3), dtype=np.uint8)
    for sq in grid.flat:
        img = sq.image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[0] != cell_px:
            img = cv2.resize(img, (cell_px, cell_px), interpolation=cv2.INTER_AREA)
        y0, x0 = sq.row * cell_px, sq.col * cell_px
        board[y0 : y0 + cell_px, x0 : x0 + cell_px] = img
    return board
