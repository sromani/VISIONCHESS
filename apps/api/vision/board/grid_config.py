"""Grid extraction configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GridExtractorConfig:
    """Parameters for splitting a warped board into 64 square crops."""

    margin_ratio: float = 0.08
    min_crop_px: int = 8
    edge_density_ratio: float = 0.18
    min_edge_density: float = 2.0
    fallback_margin_ratio: float = 0.04
    edge_snap_ratio: float = 0.14
    profile_smooth_sigma: float = 2.5
    # Super-resolution: upscale rectified board before uniform square split
    upscale_enabled: bool = True
    upscale_size: int = 2048
