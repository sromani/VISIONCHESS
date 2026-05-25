"""Multi-stage decode: occupied → color → piece type."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision.classification.confidence import ConfidenceReport, analyze_probs, softmax
from vision.classification.labels import (
    BLACK_SLICE,
    CLASS_NAMES,
    EMPTY_IDX,
    IDX_TO_CLASS,
    WHITE_SLICE,
)


@dataclass(frozen=True, slots=True)
class StagedPrediction:
    label: str
    occupied: bool
    color: str | None
    piece_kind: str | None
    occupancy_prob: float
    color_prob: float
    piece_prob: float
    confidence: ConfidenceReport


def decode_staged(
    logits: NDArray[np.float32],
    *,
    empty_threshold: float = 0.52,
) -> StagedPrediction:
    """Decode 13-class logits through three stable stages."""
    probs = softmax(logits.astype(np.float32))

    p_empty = float(probs[EMPTY_IDX])
    p_white = float(probs[WHITE_SLICE].sum())
    p_black = float(probs[BLACK_SLICE].sum())
    p_occupied = p_white + p_black

    if p_empty >= empty_threshold and p_empty >= p_occupied:
        conf = analyze_probs(probs)
        return StagedPrediction(
            label="empty",
            occupied=False,
            color=None,
            piece_kind=None,
            occupancy_prob=p_empty,
            color_prob=0.0,
            piece_prob=0.0,
            confidence=conf,
        )

    is_white = p_white >= p_black
    color_slice = WHITE_SLICE if is_white else BLACK_SLICE
    color_prob = p_white if is_white else p_black
    color = "white" if is_white else "black"

    local = probs[color_slice]
    local_idx = int(local.argmax())
    global_idx = (1 if is_white else 7) + local_idx
    label = IDX_TO_CLASS[global_idx]
    piece_kind = label.split("_", 1)[1]

    staged_probs = np.zeros_like(probs)
    staged_probs[global_idx] = float(local.max())
    conf = analyze_probs(staged_probs)

    return StagedPrediction(
        label=label,
        occupied=True,
        color=color,
        piece_kind=piece_kind,
        occupancy_prob=p_occupied,
        color_prob=color_prob,
        piece_prob=float(local.max()),
        confidence=conf,
    )


def decode_batch_staged(logits: NDArray[np.float32]) -> list[StagedPrediction]:
    return [decode_staged(row) for row in logits]
