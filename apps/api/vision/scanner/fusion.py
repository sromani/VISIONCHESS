"""Soft fusion of occupancy prior + piece classifier — no hard occupancy gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FusionMode = Literal["soft_weighted", "product", "hard_gate"]


@dataclass(frozen=True, slots=True)
class FusionConfig:
    mode: FusionMode = "soft_weighted"
    alpha: float = 0.85  # piece classifier weight
    beta: float = 0.15  # occupancy prior weight
    fused_empty_threshold: float = 0.35
    hard_gate_threshold: float = 0.65  # used only for A/B experiment


@dataclass(frozen=True, slots=True)
class SquareFusionResult:
    square_name: str
    occupancy_probability: float
    empty_probability: float
    piece_label: str
    piece_confidence: float
    fused_confidence: float
    label: str
    confidence: float
    occupied: bool
    fusion_mode: str
    hard_gate_label: str
    hard_gate_occupied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "square_name": self.square_name,
            "occupancy_probability": self.occupancy_probability,
            "empty_probability": self.empty_probability,
            "piece_label": self.piece_label,
            "piece_confidence": self.piece_confidence,
            "fused_confidence": self.fused_confidence,
            "label": self.label,
            "confidence": self.confidence,
            "occupied": self.occupied,
            "fusion_mode": self.fusion_mode,
            "hard_gate_label": self.hard_gate_label,
            "hard_gate_occupied": self.hard_gate_occupied,
        }


def fuse_confidence(piece_confidence: float, occupancy_probability: float, cfg: FusionConfig) -> float:
    piece = float(max(0.0, min(1.0, piece_confidence)))
    occ = float(max(0.0, min(1.0, occupancy_probability)))
    if cfg.mode == "product":
        return piece * occ
    if cfg.mode == "hard_gate":
        return piece if occ > cfg.hard_gate_threshold else 0.0
    # soft_weighted — occupancy is a prior, not a veto
    return cfg.alpha * piece + cfg.beta * occ


def hard_gate_decision(
    piece_label: str,
    piece_confidence: float,
    occupancy_probability: float,
    *,
    threshold: float,
) -> tuple[str, bool]:
    if occupancy_probability > threshold:
        return piece_label, True
    return "empty", False


def fuse_square(
    *,
    square_name: str,
    piece_label: str,
    piece_confidence: float,
    occupancy_probability: float,
    cfg: FusionConfig,
) -> SquareFusionResult:
    occ = float(max(0.0, min(1.0, occupancy_probability)))
    empty_prob = float(1.0 - occ)
    fused = fuse_confidence(piece_confidence, occ, cfg)

    hard_label, hard_occupied = hard_gate_decision(
        piece_label,
        piece_confidence,
        occ,
        threshold=cfg.hard_gate_threshold,
    )

    if fused >= cfg.fused_empty_threshold and piece_label != "empty":
        final_label = piece_label
        occupied = True
    else:
        final_label = "empty"
        occupied = False

    return SquareFusionResult(
        square_name=square_name,
        occupancy_probability=occ,
        empty_probability=empty_prob,
        piece_label=piece_label,
        piece_confidence=float(piece_confidence),
        fused_confidence=fused,
        label=final_label,
        confidence=fused if occupied else empty_prob,
        occupied=occupied,
        fusion_mode=cfg.mode,
        hard_gate_label=hard_label,
        hard_gate_occupied=hard_occupied,
    )


def compare_fusion_strategies(
    squares: list[dict[str, Any]],
    *,
    cfg: FusionConfig,
) -> dict[str, Any]:
    """A/B: hard occupancy gate vs active soft fusion strategy."""
    hard_cfg = FusionConfig(
        mode="hard_gate",
        hard_gate_threshold=cfg.hard_gate_threshold,
        fused_empty_threshold=cfg.fused_empty_threshold,
    )

    hard_occupied = 0
    soft_occupied = 0
    recovered: list[str] = []
    lost: list[str] = []

    for sq in squares:
        name = str(sq["square_name"])
        piece_label = str(sq.get("piece_label", sq.get("label", "unknown")))
        piece_conf = float(sq.get("piece_confidence", sq.get("confidence", 0.0)))
        occ = float(sq.get("occupancy_probability", 0.0))

        hard = fuse_square(
            square_name=name,
            piece_label=piece_label,
            piece_confidence=piece_conf,
            occupancy_probability=occ,
            cfg=hard_cfg,
        )
        soft = fuse_square(
            square_name=name,
            piece_label=piece_label,
            piece_confidence=piece_conf,
            occupancy_probability=occ,
            cfg=cfg,
        )

        if hard.occupied:
            hard_occupied += 1
        if soft.occupied:
            soft_occupied += 1
        if not hard.occupied and soft.occupied:
            recovered.append(name)
        if hard.occupied and not soft.occupied:
            lost.append(name)

    return {
        "hard_gate": {
            "strategy": "occupancy_hard_gate",
            "threshold": cfg.hard_gate_threshold,
            "occupied_count": hard_occupied,
            "empty_count": 64 - hard_occupied,
        },
        "soft_fusion": {
            "strategy": cfg.mode,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "fused_empty_threshold": cfg.fused_empty_threshold,
            "occupied_count": soft_occupied,
            "empty_count": 64 - soft_occupied,
        },
        "delta": {
            "pieces_recovered_by_soft": recovered,
            "pieces_lost_by_soft": lost,
            "recovered_count": len(recovered),
            "lost_count": len(lost),
        },
    }
