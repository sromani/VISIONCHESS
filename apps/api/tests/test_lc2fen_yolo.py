"""Tests for LC2FEN geometry + YOLO piece pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from vision.inference.yolo_detector import YoloPieceDetection
from vision.lc2fen.yolo_pieces import YOLO_TO_FEN, _fen_from_assignments, _resolve_assignments

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


def test_fen_from_sparse_yolo_assignments() -> None:
    assigned = {
        "e1": YoloPieceDetection(label="white_king", confidence=0.9, bbox=(0, 0, 10, 10), square_name="e1"),
        "e8": YoloPieceDetection(label="black_king", confidence=0.88, bbox=(0, 0, 10, 10), square_name="e8"),
    }
    fen = _fen_from_assignments(assigned)
    assert "K" in fen.split("/")[7]
    assert "k" in fen.split("/")[0]


def test_one_piece_per_square_tiebreak() -> None:
    board_size = 800
    dets = [
        YoloPieceDetection(label="white_pawn", confidence=0.4, bbox=(100, 700, 80, 80)),
        YoloPieceDetection(label="white_knight", confidence=0.7, bbox=(110, 710, 80, 80)),
    ]
    assigned = _resolve_assignments(dets, board_size)
    assert len(assigned) == 1
    square, winner = next(iter(assigned.items()))
    assert winner.label == "white_knight"
    assert square


@pytest.mark.skipif(not TEST_IMAGE.is_file(), reason="LC2FEN test image not downloaded")
def test_lc2fen_yolo_pipeline_on_test_image() -> None:
    from vision.lc2fen.adapter import LC2FENAdapter

    adapter = LC2FENAdapter()
    job_dir = REPO_ROOT / "apps" / "api" / "uploads" / "_test_lc2fen_yolo"
    job_dir.mkdir(parents=True, exist_ok=True)
    data = TEST_IMAGE.read_bytes()
    result = adapter.predict_yolo_from_bytes(data, job_dir=job_dir / "run", conf_threshold=0.25)
    assert result.fen
    assert result.metadata["assigned_count"] >= 5
    assert result.metadata["piece_detector"]["localization_only"] is False
    for label in result.assigned.values():
        assert label.label in YOLO_TO_FEN
