"""Piece silhouette detection — centered foreground blobs."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def silhouette_score(crop_bgr: NDArray[np.uint8]) -> tuple[float, float]:
    """Return (silhouette_score, center_activation) in [0, 1]."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(h, w) / 5.0)
    highpass = cv2.absdiff(gray, blur)

    center_mask = np.zeros((h, w), dtype=np.uint8)
    radius = max(4, int(min(h, w) * 0.30))
    cv2.circle(center_mask, (w // 2, h // 2), radius, 255, -1)

    _, binary = cv2.threshold(highpass, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, center_mask)

    center_activation = float(np.mean(highpass[center_mask > 0] > 16)) if center_mask.any() else 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0.0, center_activation

    best_score = 0.0
    pixel_count = float(h * w)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        area_ratio = area / pixel_count
        if area_ratio < 0.04 or area_ratio > 0.38:
            continue

        cx = stats[label_id, cv2.CC_STAT_LEFT] + stats[label_id, cv2.CC_STAT_WIDTH] / 2.0
        cy = stats[label_id, cv2.CC_STAT_TOP] + stats[label_id, cv2.CC_STAT_HEIGHT] / 2.0
        dist = np.hypot(cx - w / 2, cy - h / 2) / max(min(h, w) / 2, 1.0)
        centrality = max(0.0, 1.0 - dist)

        component = (labels == label_id).astype(np.uint8) * 255
        vert_proj = np.sum(component, axis=0).astype(np.float32)
        vert_peak = float(np.max(vert_proj) / max(np.sum(vert_proj), 1.0))

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        symmetry = 0.5
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(bw, 1)
            verticality = min(aspect / 1.8, 1.0) if aspect >= 0.8 else 0.3
            mask_f = component.astype(np.float32)
            flipped = cv2.flip(mask_f, 1)
            overlap = float(np.sum(np.minimum(mask_f, flipped))) / max(float(np.sum(mask_f)), 1.0)
            symmetry = 0.5 * verticality + 0.5 * overlap

        score = 0.35 * min(area_ratio / 0.12, 1.0) + 0.25 * centrality + 0.20 * vert_peak + 0.20 * symmetry
        best_score = max(best_score, score)

    return min(best_score, 1.0), min(center_activation, 1.0)


def edge_score(features_edge_center: float, features_edge_border: float) -> float:
    """Edge activation in center relative to border — pieces add center edges."""
    delta = features_edge_center - features_edge_border * 0.65
    return float(np.clip(delta / 0.12, 0.0, 1.0))


def entropy_score(entropy: float, template_entropy: float = 2.8) -> float:
    """Occupied squares tend to have higher entropy than empty uniform cells."""
    delta = entropy - template_entropy
    return float(np.clip(delta / 1.2, 0.0, 1.0))
