"""Calibrated confidence, entropy, and uncertainty from classifier logits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32)


def softmax_batch(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def entropy(probs: NDArray[np.float32]) -> float:
    p = probs[probs > 1e-12]
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(probs: NDArray[np.float32]) -> float:
    n = probs.size
    if n <= 1:
        return 0.0
    return entropy(probs) / float(np.log(n))


def top_k(probs: NDArray[np.float32], k: int = 3) -> list[tuple[int, float]]:
    indices = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in indices]


@dataclass(frozen=True, slots=True)
class ConfidenceReport:
    probability: float
    entropy: float
    normalized_entropy: float
    uncertainty: float
    ambiguous: bool
    top3: list[tuple[int, float]]


def analyze_probs(
    probs: NDArray[np.float32],
    *,
    ambiguity_threshold: float = 0.55,
) -> ConfidenceReport:
    """Real confidence from probability mass + uncertainty from entropy."""
    top_idx = int(probs.argmax())
    prob = float(probs[top_idx])
    ent = entropy(probs)
    norm_ent = normalized_entropy(probs)
    calibrated = prob * (1.0 - norm_ent)
    top3 = top_k(probs, 3)
    gap = top3[0][1] - top3[1][1] if len(top3) > 1 else top3[0][1]
    ambiguous = norm_ent > ambiguity_threshold or gap < 0.12

    return ConfidenceReport(
        probability=calibrated,
        entropy=ent,
        normalized_entropy=norm_ent,
        uncertainty=norm_ent,
        ambiguous=ambiguous,
        top3=top3,
    )
