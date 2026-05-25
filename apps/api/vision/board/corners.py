"""Corner ordering and quadrilateral validation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def order_points(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Uses y-sort (robust for perspective views) with sum/diff fallback for
    heavily rotated quads.
    """
    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    if points.shape != (4, 2):
        msg = f"expected shape (4, 2), got {points.shape}"
        raise ValueError(msg)

    y_sorted = order_points_y_sort(points)
    if _is_valid_convex_quad(y_sorted):
        return y_sorted

    sum_diff = order_points_sum_diff(points)
    if _is_valid_convex_quad(sum_diff):
        return sum_diff

    return order_points_by_angle(points)


def order_points_y_sort(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """TL, TR, BR, BL via top/bottom y-sort and left/right x-sort."""
    idx = np.argsort(points[:, 1])
    top = points[idx[:2]]
    bottom = points[idx[2:]]

    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def order_points_sum_diff(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Classic imutils ordering: min(x+y)=TL, max(x+y)=BR, etc."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def order_points_by_angle(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Order by polar angle around centroid; rotate so TL (min x+y) is first."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    cyclic = points[np.argsort(angles)]

    start = int(np.argmin(cyclic.sum(axis=1)))
    return np.roll(cyclic, -start, axis=0).astype(np.float32)


def order_corners(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Alias kept for backward compatibility."""
    return order_points(points)


def _is_valid_convex_quad(corners: NDArray[np.float32]) -> bool:
    contour = corners.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(contour))
    if area < 100.0:
        return False
    return bool(cv2.isContourConvex(contour))


def scale_corners(
    corners: NDArray[np.float32],
    scale: float,
) -> NDArray[np.float32]:
    if scale == 1.0:
        return corners.copy()
    return (corners / scale).astype(np.float32)


def quadrilateral_aspect_ratio(corners: NDArray[np.float32]) -> float:
    """Return width/height ratio in (0, 1] — 1.0 means perfect square."""
    ordered = order_points(corners)
    width = float(
        max(
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[3]),
        )
    )
    height = float(
        max(
            np.linalg.norm(ordered[3] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1]),
        )
    )
    if width <= 0 or height <= 0:
        return 0.0
    return min(width, height) / max(width, height)


def quadrilateral_aspect_ok(corners: NDArray[np.float32], max_deviation: float) -> bool:
    ratio = quadrilateral_aspect_ratio(corners)
    return ratio >= (1.0 - max_deviation)


def _angle_cosine(a: NDArray[np.float32], b: NDArray[np.float32], c: NDArray[np.float32]) -> float:
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(abs(np.dot(v1, v2) / denom))


def quadrilateral_angles_ok(corners: NDArray[np.float32], min_cosine: float) -> bool:
    ordered = order_points(corners)
    for i in range(4):
        prev_pt = ordered[i - 1]
        curr_pt = ordered[i]
        next_pt = ordered[(i + 1) % 4]
        if _angle_cosine(prev_pt, curr_pt, next_pt) > min_cosine:
            return False
    return True
