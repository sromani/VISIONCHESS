"""Per-square feature extraction for empty-square modeling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SquareFeatures:
    mean_bgr: tuple[float, float, float]
    mean_lab: tuple[float, float, float]
    hist_l: NDArray[np.float32]
    grad_mean: float
    grad_std: float
    edge_density_center: float
    edge_density_border: float
    texture_energy: float
    fft_high_energy: float
    local_var_center: float
    local_var_border: float
    entropy: float


def is_light_square(row: int, col: int) -> bool:
    return (row + col) % 2 == 0


def extract_features(crop_bgr: NDArray[np.uint8]) -> SquareFeatures:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    ring = max(2, int(min(h, w) * 0.16))

    center_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(center_mask, (w // 2, h // 2), max(4, int(min(h, w) * 0.32)), 255, -1)
    border_mask = np.ones((h, w), dtype=np.uint8) * 255
    border_mask[ring : h - ring, ring : w - ring] = 0

    mean_bgr = tuple(float(x) for x in np.mean(crop_bgr, axis=(0, 1)))
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    mean_lab = tuple(float(x) for x in np.mean(lab, axis=(0, 1)))

    hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
    hist_l = (hist_l / max(hist_l.sum(), 1.0)).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    edges = cv2.Canny(gray, 50, 130)
    center_edges = edges[center_mask > 0]
    border_edges = edges[border_mask > 0]
    edge_density_center = float(np.mean(center_edges > 0)) if center_edges.size else 0.0
    edge_density_border = float(np.mean(border_edges > 0)) if border_edges.size else 0.0

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture_energy = float(np.var(lap))

    fft_energy = _fft_high_energy(gray)
    center_vals = gray[center_mask > 0].astype(np.float32)
    border_vals = gray[border_mask > 0].astype(np.float32)
    local_var_center = float(np.var(center_vals)) if center_vals.size else 0.0
    local_var_border = float(np.var(border_vals)) if border_vals.size else 0.0

    entropy = _gray_entropy(gray)

    return SquareFeatures(
        mean_bgr=mean_bgr,
        mean_lab=mean_lab,
        hist_l=hist_l,
        grad_mean=grad_mean,
        grad_std=grad_std,
        edge_density_center=edge_density_center,
        edge_density_border=edge_density_border,
        texture_energy=texture_energy,
        fft_high_energy=fft_energy,
        local_var_center=local_var_center,
        local_var_border=local_var_border,
        entropy=entropy,
    )


def feature_vector(features: SquareFeatures) -> NDArray[np.float32]:
    return np.array(
        [
            *features.mean_lab,
            *features.hist_l,
            features.grad_mean,
            features.grad_std,
            features.edge_density_center,
            features.edge_density_border,
            features.texture_energy,
            features.fft_high_energy,
            features.local_var_center,
            features.local_var_border,
            features.entropy,
        ],
        dtype=np.float32,
    )


def _fft_high_energy(gray: NDArray[np.uint8]) -> float:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * 0.15)
    mask = np.ones((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 0, -1)
    high = magnitude * mask
    total = float(magnitude.sum()) + 1e-6
    return float(high.sum() / total)


def _gray_entropy(gray: NDArray[np.uint8]) -> float:
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 1e-8]
    return float(-np.sum(p * np.log(p)))
