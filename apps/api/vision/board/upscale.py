"""Board super-resolution — upscale rectified view before square extraction."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def upscale_rectified_board(
    board_bgr: NDArray[np.uint8],
    target_size: int,
    *,
    interpolation: int = cv2.INTER_CUBIC,
) -> NDArray[np.uint8]:
    """Strong upscale of a canonical rectified board for higher-resolution square crops."""
    if target_size <= 0:
        return board_bgr

    image = board_bgr if board_bgr.ndim == 3 else cv2.cvtColor(board_bgr, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]

    if h == target_size and w == target_size:
        return image

    return cv2.resize(image, (target_size, target_size), interpolation=interpolation)
