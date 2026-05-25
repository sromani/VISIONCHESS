"""Tests for ML debug capture."""

from __future__ import annotations

import numpy as np

from vision.board.types import SquareCrop
from vision.inference.model_registry import resolve_occupancy_model, resolve_piece_model
from vision.inference.piece_pipeline import PieceInferencePipeline
from vision.occupancy.ml_model import MlOccupancyModel


def _square(name: str, row: int, col: int) -> SquareCrop:
    return SquareCrop(
        row=row,
        col=col,
        image=np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        bbox=(0, 0, 128, 128),
        cell_bbox=(0, 0, 128, 128),
        cell_quad=((0, 0), (128, 0), (128, 128), (0, 128)),
        crop_quad=((0, 0), (128, 0), (128, 128), (0, 128)),
    )


def test_piece_pipeline_debug_top3_and_logits() -> None:
    artifact = resolve_piece_model()
    if artifact is None:
        return
    pipeline = PieceInferencePipeline(artifact)
    squares = [_square("e4", 4, 4)]
    _, debug = pipeline.classify_squares_with_debug(squares)
    assert len(debug) == 1
    d = debug[0]
    assert len(d.top3) == 3
    assert len(d.logits) == artifact.num_classes
    assert d.onnx_input_bgr.shape[0] == artifact.image_size
    assert abs(sum(p.probability for p in d.top3) - 1.0) < 0.01 or d.top3[0].probability > 0


def test_occupancy_debug_logits_and_crop() -> None:
    artifact = resolve_occupancy_model()
    if artifact is None:
        return
    model = MlOccupancyModel.from_artifact(artifact)
    crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    d = model.predict_debug(crop)
    assert 0.0 <= d.occupied_probability <= 1.0
    assert len(d.logits) >= 1
    assert d.onnx_input_bgr.shape[0] == artifact.image_size
