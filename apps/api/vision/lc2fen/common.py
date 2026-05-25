"""Shared LC2FEN types and image helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SquarePrediction:
    name: str
    row: int
    col: int
    label: str
    confidence: float
    occupied: bool
    probabilities: list[float] = field(default_factory=list)


def find_rectified_image(tmp_dir: Path, original_name: str) -> Path | None:
    candidate = tmp_dir / original_name
    if candidate.is_file():
        return candidate
    matches = list(tmp_dir.glob("*.jpg"))
    return matches[0] if matches else None


def warp_preview(image_bgr: np.ndarray, corners: list[list[int]]) -> np.ndarray:
    """Perspective warp for UI preview (top-down board)."""
    src = np.array(corners, dtype=np.float32)
    size = max(image_bgr.shape[0], image_bgr.shape[1], 800)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, matrix, (size, size))


def draw_corner_overlay(image_bgr: np.ndarray, corners: list[list[int]]) -> np.ndarray:
    overlay = image_bgr.copy()
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(overlay, [pts], True, (0, 220, 80), 3, cv2.LINE_AA)
    for idx, (x, y) in enumerate(corners):
        cv2.circle(overlay, (int(x), int(y)), 8, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(idx),
            (int(x) + 10, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay
