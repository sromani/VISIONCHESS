"""LC2FEN board geometry only — detection, corners, rectification."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision.lc2fen.common import find_rectified_image, warp_preview
from vision.lc2fen.bootstrap import lc2fen_runtime


@dataclass
class LC2FENGeometryResult:
    corners: list[list[int]]
    original_width: int
    original_height: int
    warped_bgr: np.ndarray
    rectified_bgr: np.ndarray
    a1_pos: str


def rectify_board(image_path: Path, *, a1_pos: str = "BL") -> LC2FENGeometryResult:
    """Run LC2FEN board detection + rectification without piece classification."""
    image_path = image_path.resolve()
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Unable to read image: {image_path}")

    with lc2fen_runtime():
        from lc2fen.predict_board import detect_input_board

        corners = detect_input_board(str(image_path), None)
        tmp_dir = image_path.parent / "tmp"
        rectified_path = find_rectified_image(tmp_dir, image_path.name)
        rectified_bgr = cv2.imread(str(rectified_path)) if rectified_path else original.copy()
        if rectified_bgr is None:
            rectified_bgr = original.copy()

    warped = warp_preview(original, corners)
    return LC2FENGeometryResult(
        corners=corners,
        original_width=original.shape[1],
        original_height=original.shape[0],
        warped_bgr=warped,
        rectified_bgr=rectified_bgr,
        a1_pos=a1_pos,
    )


def rectify_board_from_bytes(
    data: bytes,
    *,
    job_dir: Path,
    filename: str = "input.jpg",
    a1_pos: str = "BL",
) -> LC2FENGeometryResult:
    job_dir.mkdir(parents=True, exist_ok=True)
    image_path = job_dir / filename
    image_path.write_bytes(data)
    try:
        return rectify_board(image_path, a1_pos=a1_pos)
    except Exception:
        tmp_dir = job_dir / "tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
