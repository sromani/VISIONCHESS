"""Perspective-correct extraction of individual square crops."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def warp_quad_to_square(
    image: NDArray[np.uint8],
    quad: NDArray[np.float64],
    output_size: int,
) -> NDArray[np.uint8]:
    """Warp a source quadrilateral to a square crop via homography."""
    if output_size < 1:
        msg = f"output_size must be >= 1, got {output_size}"
        raise ValueError(msg)

    src = np.float32(quad)
    max_coord = float(output_size - 1)
    dst = np.float32(
        [
            [0.0, 0.0],
            [max_coord, 0.0],
            [max_coord, max_coord],
            [0.0, max_coord],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def quad_output_size(quad: NDArray[np.float64]) -> int:
    """Estimate square output size from average edge length of the quad."""
    top = float(np.linalg.norm(quad[1] - quad[0]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    return max(1, int(round((top + bottom + left + right) / 4.0)))
