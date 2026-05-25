"""Quadrilateral candidate extraction and scoring."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.config import BoardDetectorConfig
from vision.board.corners import (
    order_points,
    quadrilateral_angles_ok,
    quadrilateral_aspect_ok,
    quadrilateral_aspect_ratio,
)


@dataclass(frozen=True, slots=True)
class QuadCandidate:
    corners: NDArray[np.float32]
    score: float
    area_ratio: float
    aspect_ratio: float
    contour: NDArray[np.int32]


def find_contours(edges: NDArray[np.uint8]) -> list[NDArray[np.int32]]:
    """Return all contours — not only external — to catch inner board borders."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def _contour_area(contour: NDArray[np.int32]) -> float:
    return float(cv2.contourArea(contour))


def _approx_quads(
    contour: NDArray[np.int32],
    epsilon_ratios: tuple[float, ...],
) -> list[NDArray[np.float32]]:
    quads: list[NDArray[np.float32]] = []
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return quads

    for ratio in epsilon_ratios:
        approx = cv2.approxPolyDP(contour, ratio * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        if not any(np.allclose(quad, existing) for existing in quads):
            quads.append(quad)
    return quads


def _parallel_score(ordered: NDArray[np.float32]) -> float:
    width_top = float(np.linalg.norm(ordered[1] - ordered[0]))
    width_bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    height_left = float(np.linalg.norm(ordered[3] - ordered[0]))
    height_right = float(np.linalg.norm(ordered[2] - ordered[1]))

    return 1.0 - min(
        abs(width_top - width_bottom) / max(width_top, width_bottom, 1.0),
        abs(height_left - height_right) / max(height_left, height_right, 1.0),
    )


def _area_sweetness(area_ratio: float) -> float:
    """Peak score when the board occupies ~35–50% of the frame."""
    target = 0.42
    deviation = abs(area_ratio - target) / target
    return max(0.0, 1.0 - deviation)


def _score_quad(
    corners: NDArray[np.float32],
    image_area: float,
    config: BoardDetectorConfig,
) -> float:
    area = float(cv2.contourArea(corners.reshape(-1, 1, 2)))
    if area <= 0:
        return 0.0

    area_ratio = area / image_area
    if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
        return 0.0

    if not quadrilateral_aspect_ok(corners, config.max_aspect_ratio_deviation):
        return 0.0
    if not quadrilateral_angles_ok(corners, config.min_cosine_angle):
        return 0.0

    squareness = quadrilateral_aspect_ratio(corners)
    parallel = _parallel_score(order_points(corners))
    area_score = _area_sweetness(area_ratio)

    return float(0.45 * squareness + 0.35 * area_score + 0.20 * parallel)


def find_quad_candidates(
    edges: NDArray[np.uint8],
    config: BoardDetectorConfig,
) -> list[QuadCandidate]:
    """Return all valid quadrilateral candidates sorted by score (best first)."""
    height, width = edges.shape[:2]
    image_area = float(height * width)
    epsilons = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)

    candidates: list[QuadCandidate] = []
    seen: set[tuple[tuple[float, float], ...]] = set()

    for contour in find_contours(edges):
        area = _contour_area(contour)
        if area < image_area * config.min_area_ratio:
            continue
        if area > image_area * config.max_area_ratio:
            continue

        for quad in _approx_quads(contour, epsilons):
            key = tuple((round(float(x), 1), round(float(y), 1)) for x, y in quad)
            if key in seen:
                continue
            seen.add(key)

            score = _score_quad(quad, image_area, config)
            if score <= 0:
                continue

            ordered = order_points(quad)
            quad_area = float(cv2.contourArea(ordered.reshape(-1, 1, 2)))
            candidates.append(
                QuadCandidate(
                    corners=ordered,
                    score=score,
                    area_ratio=quad_area / image_area,
                    aspect_ratio=quadrilateral_aspect_ratio(ordered),
                    contour=contour,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def find_best_quadrilateral(
    edges: NDArray[np.uint8],
    config: BoardDetectorConfig,
) -> tuple[NDArray[np.float32], float] | None:
    candidates = find_quad_candidates(edges, config)
    if not candidates:
        return None

    best = candidates[0]
    if best.score < config.min_score:
        return None

    return best.corners, best.score
