"""Phase 4 — soft occupancy on enhanced high-res crops."""

from __future__ import annotations

from vision.occupancy.debug import render_occupancy_debug, render_occupancy_detail_panel
from vision.occupancy.detector import OccupancyDetector
from vision.scanner.context import ScanContext


def run_occupancy(ctx: ScanContext) -> dict[str, float]:
    """Compute soft occupancy probabilities from analysis-quality crops."""
    if ctx.analysis_grid is None:
        msg = "Crop quality must run before occupancy"
        raise ValueError(msg)

    grid = ctx.analysis_grid
    detector = OccupancyDetector(ctx.config.occupancy)
    capture_ml = ctx.config.collect_debug and ctx.config.ml.capture_ml_debug
    report = detector.detect_grid(grid, capture_ml_debug=capture_ml)

    if report.ml_debug:
        from vision.scanner.stages.ml_debug import store_occupancy_ml_debug

        store_occupancy_ml_debug(ctx, report.ml_debug)

    ctx.occupancy = report.results
    scores = {name: r.probability for name, r in report.results.items()}

    ctx.metadata["occupancy"] = {
        "input_grid": "analysis_high_res",
        "input_square_px": grid.square_size,
        "occupied_count": report.occupied_count,
        "empty_count": report.empty_count,
        "board_prior_applied": report.board_prior_applied,
        "board_threshold": report.board_threshold,
        "squares": [d.to_dict() for d in report.debug_rows],
    }

    if ctx.config.collect_debug:
        ctx.add_debug("occupancy", render_occupancy_debug(grid, report))
        ctx.add_debug("occupancy_detail", render_occupancy_detail_panel(report.debug_rows, board_threshold=report.board_threshold))

    return scores
