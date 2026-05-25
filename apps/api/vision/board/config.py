"""Tunable parameters for board detection."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoardDetectorConfig:
    """Configuration for ``BoardDetector``.

    Detection runs on a downscaled copy for speed; corners are mapped back
    to the original resolution before warping.
    """

    output_size: int = 800
    max_detection_dim: int = 1200

    # Preprocessing
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gaussian_kernel: int = 5

    # Canny (sigma = 0.33 is a common auto-threshold heuristic)
    canny_sigma: float = 0.33
    canny_ratio_low: float = 0.5
    canny_ratio_high: float = 1.0

    # Morphology
    dilate_iterations: int = 2
    dilate_kernel_size: int = 3

    # Contour filtering (fraction of image area)
    min_area_ratio: float = 0.05
    max_area_ratio: float = 0.95
    approx_epsilon_ratio: float = 0.02
    min_score: float = 0.35

    # Quadrilateral quality
    max_aspect_ratio_deviation: float = 0.35
    min_cosine_angle: float = 0.25  # ~75° minimum interior angle
