"""Tests for soft occupancy calibration."""

from __future__ import annotations

import numpy as np

from vision.occupancy.calibration import (
    adaptive_board_threshold,
    assign_occupied_soft,
    fuse_soft_probability,
)


class TestSoftCalibration:
    def test_fuse_returns_soft_probability_not_binary(self) -> None:
        prob = fuse_soft_probability(0.3, 0.25, 0.1, 0.05, 0.2, 0.45, ml_weight=0.5)
        assert 0.0 < prob < 1.0
        assert prob != 0.0 and prob != 1.0

    def test_empty_board_low_threshold(self) -> None:
        probs = {f"s{i}": 0.05 + i * 0.002 for i in range(64)}
        t = adaptive_board_threshold(list(probs.values()), target_pieces=24, soft_floor=0.16)
        assert t >= 0.16
        flags, _cutoff = assign_occupied_soft(probs, target_pieces=24, soft_floor=0.16)
        assert sum(flags.values()) <= 12

    def test_piece_board_selects_cluster(self) -> None:
        probs = {f"s{i}": 0.08 for i in range(48)}
        for i in range(48, 64):
            probs[f"s{i}"] = 0.55 + (i - 48) * 0.01
        flags, cutoff = assign_occupied_soft(probs, target_pieces=24, soft_floor=0.16)
        assert sum(flags.values()) >= 12
        assert cutoff < 0.55

    def test_no_hard_ml_cutoff(self) -> None:
        low_heuristic = fuse_soft_probability(0.1, 0.1, 0.05, 0.02, 0.05, 0.65)
        assert low_heuristic > 0.25
