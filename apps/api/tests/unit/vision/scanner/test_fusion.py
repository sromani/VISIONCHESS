"""Tests for occupancy + piece soft fusion."""

from __future__ import annotations

from vision.scanner.fusion import FusionConfig, compare_fusion_strategies, fuse_square


def test_soft_fusion_recovers_low_occupancy_high_piece() -> None:
    """Bishop with occ=0.35, piece=0.90 should pass soft fusion but fail hard gate."""
    cfg = FusionConfig(mode="soft_weighted", alpha=0.85, beta=0.15, fused_empty_threshold=0.35)
    result = fuse_square(
        square_name="c1",
        piece_label="white_bishop",
        piece_confidence=0.90,
        occupancy_probability=0.35,
        cfg=cfg,
    )
    assert result.fused_confidence > 0.35
    assert result.label == "white_bishop"
    assert result.occupied is True
    assert result.hard_gate_occupied is False


def test_soft_fusion_empty_when_both_low() -> None:
    cfg = FusionConfig(fused_empty_threshold=0.35)
    result = fuse_square(
        square_name="e4",
        piece_label="black_pawn",
        piece_confidence=0.25,
        occupancy_probability=0.15,
        cfg=cfg,
    )
    assert result.label == "empty"
    assert result.occupied is False


def test_compare_strategies_reports_recovered() -> None:
    cfg = FusionConfig(hard_gate_threshold=0.65, fused_empty_threshold=0.35)
    squares = [
        {
            "square_name": "c1",
            "piece_label": "white_bishop",
            "piece_confidence": 0.88,
            "occupancy_probability": 0.40,
        },
        {
            "square_name": "e4",
            "piece_label": "black_pawn",
            "piece_confidence": 0.20,
            "occupancy_probability": 0.10,
        },
    ]
    report = compare_fusion_strategies(squares, cfg=cfg)
    assert report["hard_gate"]["occupied_count"] == 0
    assert report["soft_fusion"]["occupied_count"] == 1
    assert "c1" in report["delta"]["pieces_recovered_by_soft"]
