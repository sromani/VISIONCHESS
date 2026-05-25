"""Heuristic piece classifier using color + silhouette features (no ML)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.classification.empty import EmptyDetectionConfig, detect_occupancy


@dataclass(frozen=True, slots=True)
class HeuristicClassifierConfig:
    min_confidence: float = 0.38
    empty_config: EmptyDetectionConfig = EmptyDetectionConfig()


def classify_piece_heuristic(
    crop_bgr: NDArray[np.uint8],
    row: int,
    col: int,
    config: HeuristicClassifierConfig | None = None,
    *,
    skip_occupancy: bool = False,
) -> tuple[str, float, bool, float, str | None]:
    """Returns label, confidence, occupied, occupancy_score, empty_reason.

    Expects a dataset-quality normalized square (see ``square_quality``).
    When ``skip_occupancy`` is True the caller already gated this square.
    """
    cfg = config or HeuristicClassifierConfig()
    if not skip_occupancy:
        occ = detect_occupancy(crop_bgr, crop_bgr, cfg.empty_config, row=row, col=col)
        if not occ.occupied:
            return "empty", occ.score, False, occ.score, occ.reason

    label, conf = _classify_occupied(crop_bgr, row, col)
    if conf < cfg.min_confidence:
        return "empty", conf, False, conf, "low_classifier_confidence"

    return label, conf, True, conf, None


def _classify_occupied(preprocessed_bgr: NDArray[np.uint8], row: int, col: int) -> tuple[str, float]:
    gray = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    margin = max(2, h // 8)
    corners = np.concatenate(
        [
            gray[:margin, :margin].ravel(),
            gray[:margin, -margin:].ravel(),
            gray[-margin:, :margin].ravel(),
            gray[-margin:, -margin:].ravel(),
        ]
    )
    bg = float(np.median(corners))

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(mask)) > 127:
        mask = cv2.bitwise_not(mask)

    piece_pixels = gray[mask > 0]
    if piece_pixels.size < 20:
        return "empty", 0.3

    piece_mean = float(np.mean(piece_pixels))
    is_white_piece = piece_mean > bg + 8.0
    color = "white" if is_white_piece else "black"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return f"{color}_pawn", 0.42

    cnt = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(cnt)) / float(h * w)
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect = bw / max(bh, 1)
    hull = cv2.convexHull(cnt)
    solidity = float(cv2.contourArea(cnt)) / max(float(cv2.contourArea(hull)), 1.0)

    piece_type, type_conf = _infer_piece_type(area_ratio, aspect, solidity, row, color)

    color_conf = min(1.0, abs(piece_mean - bg) / 40.0)
    confidence = 0.45 * type_conf + 0.35 * color_conf + 0.20 * min(area_ratio * 3, 1.0)
    return f"{color}_{piece_type}", min(confidence, 0.88)


def _infer_piece_type(
    area_ratio: float,
    aspect: float,
    solidity: float,
    row: int,
    color: str,
) -> tuple[str, float]:
    """Infer piece type from silhouette heuristics."""
    # Kings / queens — large central mass
    if area_ratio > 0.52 and solidity > 0.82:
        if row in (0, 7):
            return "king", 0.62
        return "queen", 0.55

    # Rooks — blocky, high solidity
    if area_ratio > 0.42 and solidity > 0.88 and 0.65 < aspect < 1.35:
        return "rook", 0.58

    # Knights — low solidity (irregular)
    if area_ratio > 0.38 and solidity < 0.78:
        return "knight", 0.52

    # Bishops — medium area, tapered top
    if area_ratio > 0.40 and 0.78 < solidity < 0.90:
        return "bishop", 0.50

    # Pawns — small or on advanced ranks
    if area_ratio < 0.38 or row in (1, 2, 5, 6):
        return "pawn", 0.56

    return "pawn", 0.45
