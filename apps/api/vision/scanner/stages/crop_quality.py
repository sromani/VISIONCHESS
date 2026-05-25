"""Phase 3b — maximize square crop quality before any piece detection."""

from __future__ import annotations

import cv2
import numpy as np

from vision.board.types import BoardGridResult
from vision.classification.square_quality import enhance_grid_for_analysis, normalize_grid_for_dataset
from vision.scanner.context import ScanContext


def run_crop_quality(ctx: ScanContext) -> BoardGridResult:
    """Enhance high-res crops; ML downscales only at inference time."""
    if ctx.raw_grid is None:
        msg = "Extraction must run before crop quality"
        raise ValueError(msg)

    cfg = ctx.config.dataset_square
    analysis = enhance_grid_for_analysis(ctx.raw_grid, cfg)
    ctx.analysis_grid = analysis

    if analysis.square_size < cfg.min_analysis_px:
        ctx.metadata.setdefault("warnings", []).append(
            f"analysis_square_px={analysis.square_size} below min={cfg.min_analysis_px}",
        )

    ctx.metadata["crop_quality"] = {
        "analysis_square_px": analysis.square_size,
        "raw_square_px": ctx.raw_grid.square_size,
        "min_required_px": cfg.min_analysis_px,
        "enhancements": ["border_shave", "clahe", "center_focus", "bilateral", "unsharp"],
        "downscale_deferred_to": "ml_inference",
    }

    if ctx.config.collect_debug:
        preview_px = min(128, analysis.square_size)
        ctx.add_debug("crop_quality", _montage(analysis, cell_px=preview_px))

    return analysis


def build_dataset_grid(ctx: ScanContext) -> BoardGridResult | None:
    """Optional 64px export grid — not used for occupancy/classification."""
    if ctx.analysis_grid is None:
        return None
    ctx.dataset_grid = normalize_grid_for_dataset(ctx.analysis_grid, ctx.config.dataset_square)
    return ctx.dataset_grid


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
