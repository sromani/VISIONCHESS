"""Tests for calibrated confidence metrics."""

from __future__ import annotations

import numpy as np

from vision.classification.confidence import analyze_probs, entropy, softmax


class TestConfidence:
    def test_softmax_sums_to_one(self) -> None:
        logits = np.array([2.0, 1.0, 0.5], dtype=np.float32)
        probs = softmax(logits)
        assert abs(float(probs.sum()) - 1.0) < 1e-5

    def test_high_confidence_low_entropy(self) -> None:
        probs = np.array([0.95, 0.03, 0.02], dtype=np.float32)
        report = analyze_probs(probs)
        assert report.probability > 0.7
        assert report.normalized_entropy < 0.3
        assert not report.ambiguous

    def test_uniform_high_entropy_ambiguous(self) -> None:
        n = 13
        probs = np.full(n, 1.0 / n, dtype=np.float32)
        report = analyze_probs(probs)
        assert report.ambiguous
        assert report.normalized_entropy > 0.9
        assert entropy(probs) > 2.0
