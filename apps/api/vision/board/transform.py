"""Perspective correction from the 8×8 playing mesh — not the photo border."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.corners import order_points
from vision.board.exceptions import InvalidCornersError
from vision.board.types import SQUARES_PER_SIDE

NUM_LINES = SQUARES_PER_SIDE + 1


def destination_square(size: int) -> NDArray[np.float32]:
    """Legacy 4-corner destination (outer mesh corners)."""
    max_coord = float(size - 1)
    return np.array(
        [[0.0, 0.0], [max_coord, 0.0], [max_coord, max_coord], [0.0, max_coord]],
        dtype=np.float32,
    )


def destination_intersection_grid(size: int) -> NDArray[np.float32]:
    """Perfect 9×9 mesh on the output square — row 0 = rank 8, col 0 = file a."""
    step = float(size - 1) / SQUARES_PER_SIDE
    grid = np.zeros((NUM_LINES, NUM_LINES, 2), dtype=np.float32)
    for row in range(NUM_LINES):
        for col in range(NUM_LINES):
            grid[row, col] = (col * step, row * step)
    return grid


def scale_intersections(
    intersections: NDArray[np.float64],
    scale: float,
) -> NDArray[np.float64]:
    """Map intersection grid from detection scale back to full-resolution coordinates."""
    if scale == 1.0:
        return intersections.copy()
    scaled = intersections.copy()
    scaled[:, :, 0] /= scale
    scaled[:, :, 1] /= scale
    return scaled


def compute_mesh_homography(
    intersections: NDArray[np.float64],
    output_size: int,
    *,
    ransac_threshold: float = 4.0,
) -> tuple[NDArray[np.float64], float]:
    """Fit homography from the full 9×9 playing mesh (81 point pairs).

    Aligns the internal square grid — not just a photo bounding quadrilateral.
    """
    src = intersections.reshape(-1, 2).astype(np.float32)
    dst = destination_intersection_grid(output_size).reshape(-1, 2).astype(np.float32)

    if src.shape[0] < 4:
        msg = "Need at least 4 mesh points for homography"
        raise InvalidCornersError(msg)

    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if matrix is None:
        msg = "OpenCV failed to compute mesh homography"
        raise InvalidCornersError(msg)

    error = _reprojection_rmse(src, dst, matrix, mask)
    return matrix, error


def compute_inner_mesh_homography(
    inner_corners: NDArray[np.float64],
    output_size: int,
    *,
    ransac_threshold: float = 3.0,
) -> tuple[NDArray[np.float64], float]:
    """Fit homography from 7×7 inner mesh intersections (49 point pairs).

    Used when OpenCV chessboard corners are available — these ARE the internal
    grid crossings; no extrapolation to a photo border.
    """
    if inner_corners.shape != (7, 7, 2):
        msg = f"Expected inner corners shape (7, 7, 2), got {inner_corners.shape}"
        raise InvalidCornersError(msg)

    step = float(output_size - 1) / SQUARES_PER_SIDE
    dst_inner = np.zeros((7, 7, 2), dtype=np.float32)
    for row in range(7):
        for col in range(7):
            dst_inner[row, col] = ((col + 1) * step, (row + 1) * step)

    src = inner_corners.reshape(-1, 2).astype(np.float32)
    dst = dst_inner.reshape(-1, 2).astype(np.float32)

    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if matrix is None:
        msg = "OpenCV failed to compute inner-mesh homography"
        raise InvalidCornersError(msg)

    error = _reprojection_rmse(src, dst, matrix, mask)
    return matrix, error


def compute_homography(
    src_corners: NDArray[np.float32],
    output_size: int,
) -> NDArray[np.float64]:
    """Four-corner homography (legacy fallback). Prefer ``compute_mesh_homography``."""
    ordered = order_points(src_corners)
    dst = destination_square(output_size)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    if matrix is None:
        raise InvalidCornersError("OpenCV failed to compute homography")
    return matrix


def mesh_corners_from_intersections(intersections: NDArray[np.float64]) -> NDArray[np.float32]:
    """Outer playing-area corners a8, h8, h1, a1 derived from the mesh."""
    a8 = intersections[0, 0]
    h8 = intersections[0, NUM_LINES - 1]
    h1 = intersections[NUM_LINES - 1, NUM_LINES - 1]
    a1 = intersections[NUM_LINES - 1, 0]
    return order_points(np.array([a8, h8, h1, a1], dtype=np.float32))


def warp_board(
    image: NDArray[np.uint8],
    homography: NDArray[np.float64],
    output_size: int,
) -> NDArray[np.uint8]:
    return cv2.warpPerspective(
        image,
        homography,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _reprojection_rmse(
    src: NDArray[np.float32],
    dst: NDArray[np.float32],
    matrix: NDArray[np.float64],
    mask: NDArray[np.uint8] | None,
) -> float:
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    if mask is not None:
        inliers = mask.ravel().astype(bool)
        if np.any(inliers):
            errors = errors[inliers]
    return float(np.sqrt(np.mean(errors**2)))
