"""Solve observed lattice → canonical constrained grid."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.exceptions import BoardNotFoundError
from vision.board.playing_grid import GridConstraintMetrics, GridConstraints, GridFrame, PlayingGrid
from vision.board.square_warp import warp_quad_to_square


@dataclass(frozen=True, slots=True)
class GridSolveResult:
    """Observed lattice fitted to a canonical playing structure."""

    observed: PlayingGrid
    canonical: PlayingGrid
    observed_metrics: GridConstraintMetrics
    canonical_metrics: GridConstraintMetrics
    cell_size: int
    reference_homography: NDArray[np.float64]

    @property
    def constraints_satisfied(self) -> bool:
        limits = GridConstraints()
        return self.canonical_metrics.satisfies(limits)


def solve_canonical(observed: PlayingGrid, output_extent: int = 800) -> GridSolveResult:
    """Project observed lattice onto a mathematically perfect canonical grid."""
    canonical = PlayingGrid.canonical(float(output_extent))
    cell_size = output_extent // 8
    ref_h = _reference_homography(observed, canonical)

    return GridSolveResult(
        observed=observed,
        canonical=canonical,
        observed_metrics=observed.metrics(),
        canonical_metrics=canonical.metrics(),
        cell_size=cell_size,
        reference_homography=ref_h,
    )


def require_observed_constraints(
    observed: PlayingGrid,
    limits: GridConstraints | None = None,
) -> GridConstraintMetrics:
    """Reject lattices that violate playing-area constraints before solve."""
    limits = limits or GridConstraints()
    metrics = observed.metrics()
    if not metrics.satisfies(limits):
        msg = f"Observed grid violates constraints: {metrics.to_dict()}"
        raise BoardNotFoundError(msg)
    return metrics


def compose_board_preview(
    image: NDArray[np.uint8],
    solve: GridSolveResult,
) -> NDArray[np.uint8]:
    """Optional debug image — geometry sampled into canonical layout, not a global warp."""
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cell = solve.cell_size
    canvas = np.zeros((cell * 8, cell * 8, 3), dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            src_quad = solve.observed.cell(row, col).quad
            patch = warp_quad_to_square(bgr, src_quad, cell)
            y0, x0 = row * cell, col * cell
            canvas[y0 : y0 + cell, x0 : x0 + cell] = patch

    return canvas


def _reference_homography(observed: PlayingGrid, canonical: PlayingGrid) -> NDArray[np.float64]:
    from vision.board.corners import order_points

    src = observed.outer_corners()
    dst = canonical.outer_corners()
    matrix = cv2.getPerspectiveTransform(src, order_points(dst))
    return matrix.astype(np.float64)
