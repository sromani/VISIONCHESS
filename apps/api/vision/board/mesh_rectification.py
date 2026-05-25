"""Grid-constrained board rectification — piecewise cell warp from 9×9 mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.corners import order_points
from vision.board.grid_intersections import cell_quad
from vision.board.square_warp import warp_quad_to_square
from vision.board.transform import destination_intersection_grid, mesh_corners_from_intersections
from vision.board.types import SQUARES_PER_SIDE

NUM_LINES = SQUARES_PER_SIDE + 1


@dataclass(frozen=True, slots=True)
class MeshQualityStats:
    """Uniformity metrics for a 9×9 intersection mesh — lower is better."""

    column_width_cv: float
    row_height_cv: float
    orthogonality_error_deg: float

    def to_dict(self) -> dict[str, float]:
        return {
            "column_width_cv": round(self.column_width_cv, 6),
            "row_height_cv": round(self.row_height_cv, 6),
            "orthogonality_error_deg": round(self.orthogonality_error_deg, 4),
        }

    @property
    def max_error(self) -> float:
        return max(self.column_width_cv, self.row_height_cv, self.orthogonality_error_deg / 90.0)


@dataclass(frozen=True, slots=True)
class MeshRectificationResult:
    """Output of grid-constrained rectification."""

    warped_image: NDArray[np.uint8]
    source_intersections: NDArray[np.float64]
    rectified_intersections: NDArray[np.float64]
    output_size: int
    cell_size: int
    source_stats: MeshQualityStats
    rectified_stats: MeshQualityStats
    reference_homography: NDArray[np.float64]

    @property
    def corners(self) -> NDArray[np.float32]:
        return mesh_corners_from_intersections(self.source_intersections)


def measure_mesh_quality(intersections: NDArray[np.float64]) -> MeshQualityStats:
    """Measure column spacing variance, row spacing variance, and orthogonality error."""
    col_widths: list[float] = []
    for col in range(SQUARES_PER_SIDE):
        widths = [
            float(np.linalg.norm(intersections[row, col + 1] - intersections[row, col]))
            for row in range(SQUARES_PER_SIDE)
        ]
        col_widths.append(float(np.mean(widths)))

    row_heights: list[float] = []
    for row in range(SQUARES_PER_SIDE):
        heights = [
            float(np.linalg.norm(intersections[row + 1, col] - intersections[row, col]))
            for col in range(SQUARES_PER_SIDE)
        ]
        row_heights.append(float(np.mean(heights)))

    mean_col = float(np.mean(col_widths)) if col_widths else 0.0
    mean_row = float(np.mean(row_heights)) if row_heights else 0.0
    col_cv = float(np.std(col_widths) / mean_col) if mean_col > 1e-6 else 0.0
    row_cv = float(np.std(row_heights) / mean_row) if mean_row > 1e-6 else 0.0

    orth_errors: list[float] = []
    for row in range(SQUARES_PER_SIDE):
        for col in range(SQUARES_PER_SIDE):
            top = intersections[row, col + 1] - intersections[row, col]
            left = intersections[row + 1, col] - intersections[row, col]
            top_len = float(np.linalg.norm(top))
            left_len = float(np.linalg.norm(left))
            if top_len < 1e-6 or left_len < 1e-6:
                continue
            cos_angle = float(np.dot(top, left) / (top_len * left_len))
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle = float(np.arccos(cos_angle))
            orth_errors.append(abs(angle - np.pi / 2.0))

    orth_deg = float(np.degrees(np.mean(orth_errors))) if orth_errors else 0.0
    return MeshQualityStats(column_width_cv=col_cv, row_height_cv=row_cv, orthogonality_error_deg=orth_deg)


def rectify_board_mesh(
    image: NDArray[np.uint8],
    intersections: NDArray[np.float64],
    output_size: int,
) -> MeshRectificationResult:
    """Rectify via 64 independent cell warps — enforces uniform 8×8 output grid."""
    if intersections.shape != (NUM_LINES, NUM_LINES, 2):
        msg = f"Expected intersections shape ({NUM_LINES}, {NUM_LINES}, 2), got {intersections.shape}"
        raise ValueError(msg)

    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cell_size = output_size // SQUARES_PER_SIDE
    canvas_size = cell_size * SQUARES_PER_SIDE
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    dst_grid = destination_intersection_grid(canvas_size).astype(np.float64)

    for row in range(SQUARES_PER_SIDE):
        for col in range(SQUARES_PER_SIDE):
            src_quad = cell_quad(intersections, row, col)
            cell_img = warp_quad_to_square(bgr, src_quad, cell_size)
            y0 = row * cell_size
            x0 = col * cell_size
            canvas[y0 : y0 + cell_size, x0 : x0 + cell_size] = cell_img

    source_stats = measure_mesh_quality(intersections)
    rectified_stats = measure_mesh_quality(dst_grid)
    reference_homography = _reference_homography(intersections, canvas_size)

    return MeshRectificationResult(
        warped_image=canvas,
        source_intersections=intersections,
        rectified_intersections=dst_grid,
        output_size=canvas_size,
        cell_size=cell_size,
        source_stats=source_stats,
        rectified_stats=rectified_stats,
        reference_homography=reference_homography,
    )


def render_mesh_on_image(
    image: NDArray[np.uint8],
    intersections: NDArray[np.float64],
    *,
    title: str = "",
    stats: MeshQualityStats | None = None,
) -> NDArray[np.uint8]:
    """Draw 9×9 mesh lines and intersections on an image."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]

    for row in range(NUM_LINES):
        p0 = intersections[row, 0]
        p1 = intersections[row, NUM_LINES - 1]
        cv2.line(
            canvas,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    for col in range(NUM_LINES):
        p0 = intersections[0, col]
        p1 = intersections[NUM_LINES - 1, col]
        cv2.line(
            canvas,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for row in range(NUM_LINES):
        for col in range(NUM_LINES):
            x, y = intersections[row, col]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                cv2.circle(canvas, (xi, yi), 3, (200, 200, 200), -1, lineType=cv2.LINE_AA)

    if title:
        cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1, cv2.LINE_AA)

    if stats is not None:
        lines = [
            f"col CV: {stats.column_width_cv:.4f}",
            f"row CV: {stats.row_height_cv:.4f}",
            f"orth: {stats.orthogonality_error_deg:.2f} deg",
        ]
        y = 44
        for line in lines:
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
            y += 18

    return canvas


def render_mesh_quality_panel(
    source_stats: MeshQualityStats,
    rectified_stats: MeshQualityStats,
    *,
    width: int = 480,
    height: int = 200,
) -> NDArray[np.uint8]:
    """Side-by-side stats panel for debug."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    cv2.putText(panel, "MESH QUALITY", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(panel, "metric", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(panel, "source", (180, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(panel, "rectified", (300, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    rows = [
        ("col width CV", source_stats.column_width_cv, rectified_stats.column_width_cv),
        ("row height CV", source_stats.row_height_cv, rectified_stats.row_height_cv),
        ("orthogonality deg", source_stats.orthogonality_error_deg, rectified_stats.orthogonality_error_deg),
    ]
    y = 88
    for label, src, dst in rows:
        cv2.putText(panel, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        src_color = (80, 180, 255) if src > 0.05 else (80, 255, 120)
        dst_color = (80, 255, 120) if dst < 0.01 else (80, 180, 255)
        cv2.putText(panel, f"{src:.4f}", (180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, src_color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{dst:.4f}", (300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dst_color, 1, cv2.LINE_AA)
        y += 28

    cv2.putText(
        panel,
        "target: all metrics -> 0",
        (12, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    return panel


def mesh_stats_metadata(
    source: MeshQualityStats,
    rectified: MeshQualityStats,
) -> dict[str, Any]:
    return {
        "source": source.to_dict(),
        "rectified": rectified.to_dict(),
    }


def _reference_homography(
    intersections: NDArray[np.float64],
    output_size: int,
) -> NDArray[np.float64]:
    """Best-fit global homography from outer corners — metadata only."""
    src = mesh_corners_from_intersections(intersections)
    dst = destination_intersection_grid(output_size)
    dst_corners = np.array(
        [dst[0, 0], dst[0, NUM_LINES - 1], dst[NUM_LINES - 1, NUM_LINES - 1], dst[NUM_LINES - 1, 0]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, order_points(dst_corners))
    return matrix.astype(np.float64)
