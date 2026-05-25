"""Phase 2 — solve observed lattice to canonical constrained grid."""

from __future__ import annotations

import numpy as np

from vision.board.exceptions import BoardNotFoundError
from vision.board.grid_solver import GridSolveResult, compose_board_preview, solve_canonical
from vision.board.playing_grid import GridConstraints
from vision.scanner.context import ScanContext


def run_mesh_rectification(ctx: ScanContext) -> GridSolveResult:
    """Fit canonical 8×8 structure — constraints must reach zero on solved lattice."""
    if ctx.observed_grid is None:
        msg = "Localization must produce observed_grid before solve"
        raise BoardNotFoundError(msg)

    solve = solve_canonical(ctx.observed_grid, ctx.config.mesh.output_size)
    ctx.grid_solve = solve
    ctx.record_geometry()

    if not solve.canonical_metrics.satisfies(GridConstraints()):
        msg = f"Canonical solve failed constraints: {solve.canonical_metrics.to_dict()}"
        raise BoardNotFoundError(msg)

    preview = compose_board_preview(ctx.original_bgr, solve)

    if ctx.config.collect_debug:
        ctx.add_debug("mesh", _render_lattice(ctx.original_bgr, solve.observed, "observed"))
        ctx.add_debug(
            "rectified_board",
            _render_lattice(preview, solve.canonical, "canonical (constraints=0)"),
        )
        ctx.add_debug("mesh_quality", _metrics_panel(solve))

    return solve


def _render_lattice(image: np.ndarray, grid, title: str) -> np.ndarray:
    from vision.scanner.stages.localization import _render_lattice as render

    return render(image, grid, title)


def _metrics_panel(solve: GridSolveResult) -> np.ndarray:
    from vision.board.mesh_rectification import render_mesh_quality_panel

    class _Adapter:
        column_width_cv = solve.observed_metrics.column_width_cv
        row_height_cv = solve.observed_metrics.row_height_cv
        orthogonality_error_deg = solve.observed_metrics.orthogonality_error_deg

    class _Canon:
        column_width_cv = solve.canonical_metrics.column_width_cv
        row_height_cv = solve.canonical_metrics.row_height_cv
        orthogonality_error_deg = solve.canonical_metrics.orthogonality_error_deg

    return render_mesh_quality_panel(_Adapter(), _Canon())
