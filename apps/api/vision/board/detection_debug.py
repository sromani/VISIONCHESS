"""Pre-detection debug visualizations (contours, corners, homography)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.contours import QuadCandidate
from vision.board.transform import destination_square

CORNER_LABELS = ("a8", "h8", "h1", "a1")
CORNER_COLORS = (
    (80, 80, 255),   # TL — red
    (80, 255, 80),   # TR — green
    (255, 80, 80),   # BR — blue
    (80, 255, 255),  # BL — yellow
)


@dataclass(frozen=True, slots=True)
class DetectionDebugImages:
    """JPEG-ready debug frames for each pipeline step."""

    edges: NDArray[np.uint8]
    all_contours: NDArray[np.uint8]
    selected_contour: NDArray[np.uint8]
    approx_polygon: NDArray[np.uint8]
    ordered_corners: NDArray[np.uint8]
    bounding_box: NDArray[np.uint8]
    homography_points: NDArray[np.uint8]
    grid_rectification: NDArray[np.uint8]
    mesh_original: NDArray[np.uint8]
    mesh_rectified: NDArray[np.uint8]
    mesh_quality: NDArray[np.uint8]
    warped: NDArray[np.uint8]

    def as_dict(self) -> dict[str, NDArray[np.uint8]]:
        return {
            "edges": self.edges,
            "all_contours": self.all_contours,
            "selected_contour": self.selected_contour,
            "approx_polygon": self.approx_polygon,
            "ordered_corners": self.ordered_corners,
            "bounding_box": self.bounding_box,
            "homography_points": self.homography_points,
            "grid_rectification": self.grid_rectification,
            "mesh_original": self.mesh_original,
            "mesh_rectified": self.mesh_rectified,
            "mesh_quality": self.mesh_quality,
            "warped": self.warped,
        }


def _to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _scale_to_display(image: NDArray[np.uint8], max_dim: int = 900) -> NDArray[np.uint8]:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def render_edges_overlay(base: NDArray[np.uint8], edges: NDArray[np.uint8]) -> NDArray[np.uint8]:
    canvas = _to_bgr(base)
    mask = edges > 0
    canvas[mask] = (0, 255, 255)
    return _scale_to_display(canvas)


def render_all_contours(
    base: NDArray[np.uint8],
    contours: list[NDArray[np.int32]],
    candidates: list[QuadCandidate] | None = None,
) -> NDArray[np.uint8]:
    canvas = _to_bgr(base)
    cv2.drawContours(canvas, contours, -1, (120, 120, 120), 1)

    if candidates:
        for idx, cand in enumerate(candidates[:8]):
            color = (0, 200, 255) if idx == 0 else (180, 180, 180)
            thickness = 2 if idx == 0 else 1
            pts = cand.corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(canvas, [pts], True, color, thickness)
            cx = int(cand.corners[:, 0].mean())
            cy = int(cand.corners[:, 1].mean())
            cv2.putText(
                canvas,
                f"#{idx + 1} {cand.score:.2f}",
                (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return _scale_to_display(canvas)


def render_selected_contour(
    base: NDArray[np.uint8],
    contour: NDArray[np.int32] | None,
) -> NDArray[np.uint8]:
    canvas = _to_bgr(base)
    if contour is not None:
        cv2.drawContours(canvas, [contour], -1, (0, 255, 255), 2)
    return _scale_to_display(canvas)


def render_approx_polygon(
    base: NDArray[np.uint8],
    corners: NDArray[np.float32] | None,
) -> NDArray[np.uint8]:
    canvas = _to_bgr(base)
    if corners is not None:
        pts = corners.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], True, (255, 180, 0), 2)
        for pt in corners:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 5, (255, 180, 0), -1)
    return _scale_to_display(canvas)


def render_ordered_corners(
    base: NDArray[np.uint8],
    corners: NDArray[np.float32] | None,
) -> NDArray[np.uint8]:
    canvas = _to_bgr(base)
    if corners is None:
        return _scale_to_display(canvas)

    pts = corners.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (255, 255, 255), 1)

    for idx, (pt, label, color) in enumerate(zip(corners, CORNER_LABELS, CORNER_COLORS)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(canvas, (x, y), 8, color, -1)
        cv2.circle(canvas, (x, y), 8, (0, 0, 0), 1)
        cv2.putText(
            canvas,
            f"{label} ({idx})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return _scale_to_display(canvas)


def render_bounding_box(
    base: NDArray[np.uint8],
    corners: NDArray[np.float32] | None,
) -> NDArray[np.uint8]:
    canvas = _to_bgr(base)
    if corners is not None:
        x, y, w, h = cv2.boundingRect(corners.reshape(-1, 1, 2).astype(np.int32))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            canvas,
            f"{w}×{h}px",
            (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return _scale_to_display(canvas)


def render_homography_points(
    base: NDArray[np.uint8],
    src_corners: NDArray[np.float32],
    output_size: int,
) -> NDArray[np.uint8]:
    """Show src corners on original and dst square side-by-side."""
    left = render_ordered_corners(base, src_corners)
    dst = destination_square(output_size)
    size = min(left.shape[0], 480)
    square = np.full((size, size, 3), 30, dtype=np.uint8)

    scale = (size - 40) / float(output_size - 1)
    offset = 20
    dst_scaled = (dst * scale + offset).astype(np.int32)
    cv2.polylines(square, [dst_scaled.reshape(-1, 1, 2)], True, (255, 255, 255), 1)

    for idx, (pt, label, color) in enumerate(zip(dst_scaled, CORNER_LABELS, CORNER_COLORS)):
        cv2.circle(square, (int(pt[0]), int(pt[1])), 6, color, -1)
        cv2.putText(
            square,
            label,
            (int(pt[0]) + 8, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(square, "dst → warp", (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    h = max(left.shape[0], square.shape[0])
    left_pad = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    square_pad = cv2.copyMakeBorder(square, 0, h - square.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.hstack([left_pad, square_pad])


def build_detection_debug(
    *,
    original: NDArray[np.uint8],
    edges: NDArray[np.uint8],
    contours: list[NDArray[np.int32]],
    candidates: list[QuadCandidate],
    selected: QuadCandidate | None,
    warped: NDArray[np.uint8],
    output_size: int,
    grid_corners: NDArray[np.float32] | None = None,
    grid_overlay: NDArray[np.uint8] | None = None,
    mesh_original: NDArray[np.uint8] | None = None,
    mesh_rectified: NDArray[np.uint8] | None = None,
    mesh_quality: NDArray[np.uint8] | None = None,
) -> DetectionDebugImages:
    corners = grid_corners if grid_corners is not None else (selected.corners if selected else None)
    contour = selected.contour if selected else None
    fallback = _scale_to_display(_to_bgr(original))

    return DetectionDebugImages(
        edges=render_edges_overlay(original, edges),
        all_contours=render_all_contours(original, contours, candidates),
        selected_contour=render_selected_contour(original, contour),
        approx_polygon=render_approx_polygon(original, corners),
        ordered_corners=render_ordered_corners(original, corners),
        bounding_box=render_bounding_box(original, corners),
        homography_points=render_homography_points(original, corners, output_size)
        if corners is not None
        else fallback,
        grid_rectification=_scale_to_display(grid_overlay)
        if grid_overlay is not None
        else render_ordered_corners(original, corners),
        mesh_original=_scale_to_display(mesh_original) if mesh_original is not None else fallback,
        mesh_rectified=_scale_to_display(mesh_rectified) if mesh_rectified is not None else fallback,
        mesh_quality=mesh_quality if mesh_quality is not None else fallback,
        warped=_scale_to_display(_to_bgr(warped)),
    )
