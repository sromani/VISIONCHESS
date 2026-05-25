"""Production ML pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MlPipelineConfig:
    """Controls ML-first behaviour — heuristics only when explicitly allowed."""

    require_occupancy_model: bool = True
    require_piece_model: bool = True
    allow_heuristic_fallback: bool = True
    hypothesis_top_k_alternatives: int = 2
    use_stockfish_scoring: bool = True
    stockfish_depth: int = 10
    capture_ml_debug: bool = True
