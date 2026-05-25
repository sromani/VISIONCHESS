"""Chess board as a constrained 8×8 geometric structure — not an image.

The playing area is a 9×9 intersection lattice. Each of the 64 cells is a
quadrilateral bounded by four adjacent intersections. Mathematical constraints
(uniform spacing, orthogonality, parallel ranks/files) are evaluated on the
lattice itself; pixels are sampled later via ``grid_sampling``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vision.board.types import SQUARES_PER_SIDE

NUM_LINES = SQUARES_PER_SIDE + 1
CORNER_LABELS = ("a8", "h8", "h1", "a1")


class GridFrame(StrEnum):
    """Coordinate frame of the lattice."""

    IMAGE = "image"
    CANONICAL = "canonical"


@dataclass(frozen=True, slots=True)
class GridConstraints:
    """Hard limits for a valid playing grid."""

    max_column_width_cv: float = 0.35
    max_row_height_cv: float = 0.35
    max_orthogonality_deg: float = 12.0
    max_row_parallelism_deg: float = 8.0
    max_col_parallelism_deg: float = 8.0


@dataclass(frozen=True, slots=True)
class GridConstraintMetrics:
    """Measured constraint residuals — zero on a perfect canonical grid."""

    column_width_cv: float
    row_height_cv: float
    orthogonality_error_deg: float
    row_parallelism_deg: float
    col_parallelism_deg: float

    @property
    def max_error(self) -> float:
        return max(
            self.column_width_cv,
            self.row_height_cv,
            self.orthogonality_error_deg / 90.0,
            self.row_parallelism_deg / 90.0,
            self.col_parallelism_deg / 90.0,
        )

    def satisfies(self, limits: GridConstraints) -> bool:
        return (
            self.column_width_cv <= limits.max_column_width_cv
            and self.row_height_cv <= limits.max_row_height_cv
            and self.orthogonality_error_deg <= limits.max_orthogonality_deg
            and self.row_parallelism_deg <= limits.max_row_parallelism_deg
            and self.col_parallelism_deg <= limits.max_col_parallelism_deg
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "column_width_cv": round(self.column_width_cv, 6),
            "row_height_cv": round(self.row_height_cv, 6),
            "orthogonality_error_deg": round(self.orthogonality_error_deg, 4),
            "row_parallelism_deg": round(self.row_parallelism_deg, 4),
            "col_parallelism_deg": round(self.col_parallelism_deg, 4),
        }


@dataclass(frozen=True, slots=True)
class CellGeometry:
    """One board square as a geometric cell — four corners + derived metrics."""

    row: int
    col: int
    quad: NDArray[np.float64]  # TL, TR, BR, BL

    @property
    def center(self) -> tuple[float, float]:
        c = self.quad.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def width(self) -> float:
        top = float(np.linalg.norm(self.quad[1] - self.quad[0]))
        bottom = float(np.linalg.norm(self.quad[2] - self.quad[3]))
        return (top + bottom) / 2.0

    @property
    def height(self) -> float:
        left = float(np.linalg.norm(self.quad[3] - self.quad[0]))
        right = float(np.linalg.norm(self.quad[2] - self.quad[1]))
        return (left + right) / 2.0

    @property
    def square_name(self) -> str:
        rank = SQUARES_PER_SIDE - self.row
        file = chr(ord("a") + self.col)
        return f"{file}{rank}"


@dataclass(frozen=True, slots=True)
class PlayingGrid:
    """9×9 intersection lattice defining the 8×8 playing structure."""

    intersections: NDArray[np.float64]  # (9, 9, 2) — row 0 = rank 8, col 0 = file a
    frame: GridFrame = GridFrame.IMAGE

    def __post_init__(self) -> None:
        if self.intersections.shape != (NUM_LINES, NUM_LINES, 2):
            msg = f"Expected ({NUM_LINES}, {NUM_LINES}, 2), got {self.intersections.shape}"
            raise ValueError(msg)

    @classmethod
    def from_intersections(
        cls,
        intersections: NDArray[np.float64],
        *,
        frame: GridFrame = GridFrame.IMAGE,
    ) -> PlayingGrid:
        return cls(intersections=intersections.astype(np.float64), frame=frame)

    @classmethod
    def canonical(cls, extent: float = 800.0) -> PlayingGrid:
        """Perfect unit lattice on [0, extent-1]² — all constraints exactly zero."""
        step = (extent - 1.0) / SQUARES_PER_SIDE
        grid = np.zeros((NUM_LINES, NUM_LINES, 2), dtype=np.float64)
        for row in range(NUM_LINES):
            for col in range(NUM_LINES):
                grid[row, col] = (col * step, row * step)
        return cls(intersections=grid, frame=GridFrame.CANONICAL)

    def point(self, row: int, col: int) -> tuple[float, float]:
        p = self.intersections[row, col]
        return float(p[0]), float(p[1])

    def cell(self, row: int, col: int) -> CellGeometry:
        quad = np.array(
            [
                self.intersections[row, col],
                self.intersections[row, col + 1],
                self.intersections[row + 1, col + 1],
                self.intersections[row + 1, col],
            ],
            dtype=np.float64,
        )
        return CellGeometry(row=row, col=col, quad=quad)

    def all_cells(self) -> tuple[CellGeometry, ...]:
        return tuple(
            self.cell(row, col)
            for row in range(SQUARES_PER_SIDE)
            for col in range(SQUARES_PER_SIDE)
        )

    def outer_corners(self) -> NDArray[np.float32]:
        """a8, h8, h1, a1 from lattice corners."""
        a8 = self.intersections[0, 0]
        h8 = self.intersections[0, NUM_LINES - 1]
        h1 = self.intersections[NUM_LINES - 1, NUM_LINES - 1]
        a1 = self.intersections[NUM_LINES - 1, 0]
        return np.array([a8, h8, h1, a1], dtype=np.float32)

    def metrics(self) -> GridConstraintMetrics:
        return measure_constraints(self.intersections)

    def satisfies(self, limits: GridConstraints | None = None) -> bool:
        return self.metrics().satisfies(limits or GridConstraints())

    def to_metadata(self) -> dict[str, Any]:
        m = self.metrics()
        corners = self.outer_corners()
        return {
            "frame": self.frame.value,
            "constraints": m.to_dict(),
            "constraints_ok": self.satisfies(),
            "corners": {
                label: [float(corners[i, 0]), float(corners[i, 1])]
                for i, label in enumerate(CORNER_LABELS)
            },
        }


def measure_constraints(intersections: NDArray[np.float64]) -> GridConstraintMetrics:
    """Evaluate mathematical constraints on the intersection lattice."""
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
            orth_errors.append(_angle_deviation_from_90(top, left))

    row_angles = _line_family_angles(intersections, horizontal=True)
    col_angles = _line_family_angles(intersections, horizontal=False)

    return GridConstraintMetrics(
        column_width_cv=col_cv,
        row_height_cv=row_cv,
        orthogonality_error_deg=float(np.degrees(np.mean(orth_errors))) if orth_errors else 0.0,
        row_parallelism_deg=_parallelism_spread_deg(row_angles),
        col_parallelism_deg=_parallelism_spread_deg(col_angles),
    )


def _angle_deviation_from_90(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
    len1 = float(np.linalg.norm(v1))
    len2 = float(np.linalg.norm(v2))
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    cos_angle = float(np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0))
    return abs(float(np.arccos(cos_angle)) - np.pi / 2.0)


def _line_family_angles(intersections: NDArray[np.float64], *, horizontal: bool) -> list[float]:
    angles: list[float] = []
    if horizontal:
        for row in range(NUM_LINES):
            p0 = intersections[row, 0]
            p1 = intersections[row, NUM_LINES - 1]
            angles.append(float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    else:
        for col in range(NUM_LINES):
            p0 = intersections[0, col]
            p1 = intersections[NUM_LINES - 1, col]
            angles.append(float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    return angles


def _parallelism_spread_deg(angles: list[float]) -> float:
    if not angles:
        return 0.0
    mean = float(np.mean(angles))
    diffs = [abs(_wrap_angle(a - mean)) for a in angles]
    return float(np.degrees(max(diffs)))


def _wrap_angle(angle: float) -> float:
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return angle
