"""Assemble ML debug report and render debug frames."""

from __future__ import annotations

from vision.board.types import BoardGridResult
from vision.inference.ml_debug_types import MlDebugReport, SquareMlDebug
from vision.inference.model_registry import resolve_occupancy_model, resolve_piece_model
from vision.scanner.context import ScanContext
from vision.scanner.debug.ml_viz import (
    render_ml_detail_panel,
    render_ml_occupancy_heatmap,
    render_ml_piece_top1,
    render_onnx_crop_montage,
)


def run_ml_debug(ctx: ScanContext) -> MlDebugReport | None:
    if not ctx.config.collect_debug or not ctx.config.ml.capture_ml_debug:
        return None
    if ctx.analysis_grid is None:
        return None

    occ_map = _occupancy_ml_map(ctx)
    piece_map = _piece_ml_map(ctx)
    if not occ_map and not piece_map:
        return None

    piece_art = resolve_piece_model(ctx.config.classification.model_path)
    occ_art = resolve_occupancy_model(ctx.config.occupancy.ml_model_path)

    squares: dict[str, SquareMlDebug] = {}
    for sq in ctx.analysis_grid.flat:
        h, w = sq.image.shape[:2]
        squares[sq.square_name] = SquareMlDebug(
            square_name=sq.square_name,
            row=sq.row,
            col=sq.col,
            analysis_crop_shape=(h, w),
            occupancy=occ_map.get(sq.square_name),
            piece=piece_map.get(sq.square_name),
        )

    report = MlDebugReport(
        piece_model=piece_art.source if piece_art else "none",
        occupancy_model=occ_art.source if occ_art else "none",
        squares=squares,
    )
    ctx.ml_debug_report = report
    ctx.metadata["ml_debug"] = report.to_dict()

    threshold = ctx.config.occupancy.occupied_threshold
    fused_threshold = ctx.config.occupancy.fused_empty_threshold
    fusion_map = {
        s["square_name"]: s
        for s in ctx.metadata.get("piece_detection", {}).get("squares", [])
    }
    ctx.add_debug("ml_occupancy", render_ml_occupancy_heatmap(report, occupied_threshold=threshold))
    ctx.add_debug("ml_piece_top1", render_ml_piece_top1(report))
    ctx.add_debug("ml_onnx_occ_crops", render_onnx_crop_montage(report, kind="occupancy"))
    ctx.add_debug("ml_onnx_piece_crops", render_onnx_crop_montage(report, kind="piece"))
    ctx.add_debug(
        "ml_detail",
        render_ml_detail_panel(
            report,
            fusion_by_name=fusion_map,
            fused_empty_threshold=fused_threshold,
        ),
    )
    return report


def _occupancy_ml_map(ctx: ScanContext) -> dict:
    return ctx.occupancy_ml_debug_map


def _piece_ml_map(ctx: ScanContext) -> dict:
    return ctx.piece_ml_debug_map


def store_occupancy_ml_debug(ctx: ScanContext, ml_debug: dict | None) -> None:
    if ml_debug:
        ctx.occupancy_ml_debug_map = ml_debug


def store_piece_ml_debug_map(ctx: ScanContext, piece_debug_by_name: dict[str, object | None]) -> None:
    ctx.piece_ml_debug_map = {k: v for k, v in piece_debug_by_name.items() if v is not None}


def store_piece_ml_debug(ctx: ScanContext, piece_debug: list) -> None:
    """Legacy dense list — one debug row per square in grid order."""
    if not piece_debug or ctx.analysis_grid is None:
        return
    by_name: dict[str, object] = {}
    for sq, dbg in zip(ctx.analysis_grid.flat, piece_debug, strict=True):
        if dbg is not None:
            by_name[sq.square_name] = dbg
    store_piece_ml_debug_map(ctx, by_name)
