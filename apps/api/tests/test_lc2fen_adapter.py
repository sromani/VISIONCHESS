"""Tests for LiveChess2FEN integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from vision.lc2fen.adapter import LC2FENAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_IMAGE = (
    REPO_ROOT
    / "ml"
    / "vendor"
    / "LiveChess2FEN"
    / "data"
    / "predictions"
    / "TestImages"
    / "FullDetection"
    / "test1.jpg"
)


@pytest.mark.skipif(not TEST_IMAGE.is_file(), reason="LC2FEN test image not downloaded")
def test_lc2fen_predicts_fen_from_test_image():
    adapter = LC2FENAdapter()
    result = adapter.predict_from_path(TEST_IMAGE)

    assert result.fen
    assert len(result.square_predictions) == 64
    assert result.validation.is_valid
    assert result.validation.board_ready
    assert result.validation.interactive_fen
    assert result.processing_ms > 0
