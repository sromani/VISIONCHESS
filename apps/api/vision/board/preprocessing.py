"""Lighting-normalization and denoising."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.config import BoardDetectorConfig


def apply_clahe(gray: NDArray[np.uint8], config: BoardDetectorConfig) -> NDArray[np.uint8]:
    """Contrast-limited adaptive histogram equalization for uneven lighting."""
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
    )
    return clahe.apply(gray)


def denoise(gray: NDArray[np.uint8], config: BoardDetectorConfig) -> NDArray[np.uint8]:
    k = config.gaussian_kernel
    if k <= 1:
        return gray
    return cv2.GaussianBlur(gray, (k, k), 0)


def preprocess_gray(gray: NDArray[np.uint8], config: BoardDetectorConfig) -> NDArray[np.uint8]:
    normalized = apply_clahe(gray, config)
    return denoise(normalized, config)
