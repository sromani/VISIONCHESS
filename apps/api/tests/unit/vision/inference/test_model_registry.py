"""Tests for model registry and chesscog artifact metadata."""

from __future__ import annotations

from vision.inference.model_registry import resolve_occupancy_model, resolve_piece_model


def test_chesscog_piece_model_resolves() -> None:
    artifact = resolve_piece_model()
    assert artifact is not None
    assert artifact.source == "chesscog"
    assert artifact.image_size == 299
    assert artifact.num_classes == 12
    assert artifact.includes_empty_class is False
    assert len(artifact.class_names) == 12
    assert artifact.label_for_index(0) in artifact.class_names


def test_chesscog_occupancy_model_resolves() -> None:
    artifact = resolve_occupancy_model()
    assert artifact is not None
    assert artifact.source == "chesscog"
    assert artifact.image_size == 100
    assert artifact.num_classes == 2
    assert artifact.class_names == ("empty", "occupied")
