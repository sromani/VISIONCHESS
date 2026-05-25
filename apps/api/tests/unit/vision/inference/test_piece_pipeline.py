"""Tests for ML piece inference pipeline."""

from __future__ import annotations

import numpy as np

from vision.board.types import SquareCrop
from vision.inference.model_registry import resolve_piece_model
from vision.inference.piece_pipeline import PieceInferencePipeline


def test_piece_model_is_available() -> None:
    artifact = resolve_piece_model()
    assert artifact is not None
    assert artifact.path.exists()


def test_batched_inference_64_squares() -> None:
    artifact = resolve_piece_model()
    assert artifact is not None
    pipeline = PieceInferencePipeline(artifact)
    squares = [
        SquareCrop(
            row=r,
            col=c,
            image=np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            bbox=(0, 0, 128, 128),
            cell_bbox=(0, 0, 128, 128),
            cell_quad=((0, 0), (128, 0), (128, 128), (0, 128)),
            crop_quad=((0, 0), (128, 0), (128, 128), (0, 128)),
        )
        for r in range(8)
        for c in range(8)
    ]
    preds = pipeline.classify_squares(squares, soft=True)
    assert len(preds) == 64
    assert all(p.label for p in preds)
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)
