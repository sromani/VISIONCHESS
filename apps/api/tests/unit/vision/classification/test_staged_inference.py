"""Tests for multi-stage logits decode."""

from __future__ import annotations

import numpy as np

from vision.classification.labels import CLASS_NAMES, CLASS_TO_IDX
from vision.classification.staged_inference import decode_staged


def _one_hot(label: str, strength: float = 8.0) -> np.ndarray:
    logits = np.full(len(CLASS_NAMES), -4.0, dtype=np.float32)
    logits[CLASS_TO_IDX[label]] = strength
    return logits


class TestStagedInference:
    def test_empty_square(self) -> None:
        logits = _one_hot("empty")
        pred = decode_staged(logits)
        assert pred.label == "empty"
        assert not pred.occupied
        assert pred.occupancy_prob > 0.5

    def test_white_pawn(self) -> None:
        logits = _one_hot("white_pawn")
        pred = decode_staged(logits)
        assert pred.label == "white_pawn"
        assert pred.occupied
        assert pred.color == "white"
        assert pred.piece_kind == "pawn"

    def test_black_king(self) -> None:
        logits = _one_hot("black_king")
        pred = decode_staged(logits)
        assert pred.label == "black_king"
        assert pred.color == "black"
        assert pred.piece_kind == "king"

    def test_ambiguous_empty_vs_piece(self) -> None:
        logits = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        logits[CLASS_TO_IDX["empty"]] = 1.0
        logits[CLASS_TO_IDX["white_pawn"]] = 0.95
        pred = decode_staged(logits, empty_threshold=0.55)
        assert pred.occupied
