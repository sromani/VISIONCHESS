"""Edge extraction."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.config import BoardDetectorConfig
from vision.board.preprocessing import denoise


def _canny_thresholds(gray: NDArray[np.uint8], sigma: float) -> tuple[int, int]:
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return lower, upper


def detect_edges(gray: NDArray[np.uint8], config: BoardDetectorConfig) -> NDArray[np.uint8]:
    lower, upper = _canny_thresholds(gray, config.canny_sigma)
    lower = int(lower * config.canny_ratio_low)
    upper = int(max(lower + 1, upper * config.canny_ratio_high))
    return cv2.Canny(gray, lower, upper)


def strengthen_edges(edges: NDArray[np.uint8], config: BoardDetectorConfig) -> NDArray[np.uint8]:
    if config.dilate_iterations <= 0:
        return edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (config.dilate_kernel_size, config.dilate_kernel_size),
    )
    return cv2.dilate(edges, kernel, iterations=config.dilate_iterations)


def adaptive_binary_edges(gray: NDArray[np.uint8], config: BoardDetectorConfig) -> NDArray[np.uint8]:
    """Fallback for low-contrast boards under uneven lighting."""
    blurred = denoise(gray, config)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    inverted = cv2.bitwise_not(binary)
    return strengthen_edges(inverted, config)
