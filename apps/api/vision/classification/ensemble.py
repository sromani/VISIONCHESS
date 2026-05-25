"""Test-time augmentation ensemble — average logits across views."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def tta_views(rgb: NDArray[np.uint8]) -> list[NDArray[np.uint8]]:
    """Generate augmented views for ensemble inference."""
    views = [rgb]
    views.append(cv2.flip(rgb, 1))
    views.append(cv2.flip(rgb, 0))
    for angle in (-8, 8):
        h, w = rgb.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        views.append(cv2.warpAffine(rgb, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101))
    return views


def ensemble_logits(
    rgb: NDArray[np.uint8],
    infer_fn,
) -> NDArray[np.float32]:
    """Run ``infer_fn(view) -> logits`` on TTA views and average."""
    views = tta_views(rgb)
    accum: NDArray[np.float32] | None = None
    for view in views:
        logits = infer_fn(view).astype(np.float32)
        accum = logits if accum is None else accum + logits
    assert accum is not None
    return accum / len(views)
