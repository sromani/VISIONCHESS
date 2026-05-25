"""Square crop preprocessing for classification."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    output_size: int = 64
    center_crop_ratio: float = 0.72
    clahe_clip: float = 2.0
    clahe_grid: int = 4


def preprocess_square(
    crop_bgr: NDArray[np.uint8],
    config: PreprocessConfig | None = None,
) -> NDArray[np.uint8]:
    """square → grayscale → normalize → center crop → resize."""
    cfg = config or PreprocessConfig()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    normalized = clahe.apply(gray)

    h, w = normalized.shape[:2]
    side = int(min(h, w) * cfg.center_crop_ratio)
    side = max(side, 8)
    cy, cx = h // 2, w // 2
    y0 = max(cy - side // 2, 0)
    x0 = max(cx - side // 2, 0)
    y1 = min(y0 + side, h)
    x1 = min(x0 + side, w)
    center = normalized[y0:y1, x0:x1]

    if center.size == 0:
        center = normalized

    resized = cv2.resize(center, (cfg.output_size, cfg.output_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def preprocess_rgb_for_model(crop_rgb: NDArray[np.uint8], size: int = 64) -> NDArray[np.uint8]:
    """RGB crop resized for ML backends."""
    if crop_rgb.shape[0] != size or crop_rgb.shape[1] != size:
        return cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return crop_rgb
