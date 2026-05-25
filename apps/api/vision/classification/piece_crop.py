"""Extract exact piece crops from YOLO bounding boxes."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def extract_piece_crop(
    board_bgr: NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
    *,
    pad_ratio: float = 0.05,
) -> NDArray[np.uint8]:
    """Crop the piece region from the rectified board (BGR)."""
    h, w = board_bgr.shape[:2]
    x, y, bw, bh = bbox
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)
    if x1 <= x0 or y1 <= y0:
        return board_bgr[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)].copy()
    return board_bgr[y0:y1, x0:x1].copy()


def resize_crop_preview(crop: NDArray[np.uint8], size: int = 128) -> NDArray[np.uint8]:
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
