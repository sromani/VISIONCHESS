"""Occupancy detection configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OccupancyConfig:
    """Soft-probability occupancy — board-level calibration, no hard per-square cut."""

    # Fusion weights (sum ≈ 1.0)
    weight_foreground: float = 0.32
    weight_silhouette: float = 0.28
    weight_edge: float = 0.18
    weight_entropy: float = 0.12
    weight_center: float = 0.10

    # Soft calibration
    ml_weight: float = 0.55
    calibration_temperature: float = 1.15
    soft_floor: float = 0.16
    soft_ceiling: float = 0.58

    # Board-level target (typical game: 20–32 pieces)
    target_pieces: int = 24
    min_expected_pieces: int = 12
    max_expected_pieces: int = 32
    hard_max_pieces: int = 36

    # Classifier gate: include squares above this soft prob (below board occupied flag)
    classify_soft_threshold: float = 0.14

    empty_seed_fraction: float = 0.45
    ml_model_path: Path | None = None
    confidence_empty: float = 0.94
    ml_only: bool = False
    # Legacy hard gate — kept for A/B experiment comparison only
    occupied_threshold: float = 0.65
    # Soft fusion (production default for piece_detection_only)
    fusion_mode: str = "soft_weighted"  # soft_weighted | product | hard_gate
    fusion_alpha: float = 0.85
    fusion_beta: float = 0.15
    fused_empty_threshold: float = 0.35
