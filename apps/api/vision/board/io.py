"""Image loading, resizing, and encoding helpers."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def decode_image_bytes(data: bytes) -> NDArray[np.uint8]:
    """Decode JPEG/PNG/WebP bytes into a BGR image."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        msg = "Unable to decode image bytes"
        raise ValueError(msg)
    return image


def encode_jpeg(image: NDArray[np.uint8], quality: int = 90) -> bytes:
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        msg = "Failed to encode image as JPEG"
        raise ValueError(msg)
    return buffer.tobytes()


def to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_scale(original_dim: int, max_dim: int) -> float:
    if original_dim <= max_dim:
        return 1.0
    return max_dim / original_dim


def resize_for_detection(
    image: NDArray[np.uint8],
    max_dim: int,
) -> tuple[NDArray[np.uint8], float]:
    """Downscale preserving aspect ratio; return scale factor applied."""
    height, width = image.shape[:2]
    scale = min(compute_scale(width, max_dim), compute_scale(height, max_dim))
    if scale >= 1.0:
        return image, 1.0

    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale
