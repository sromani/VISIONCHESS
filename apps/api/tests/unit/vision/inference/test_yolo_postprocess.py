"""Tests for YOLO ONNX postprocessing fixes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision.inference.yolo_detector import YoloDetectorConfig, YoloPieceDetector, _postprocess
from vision.inference.model_registry import resolve_yolo_model


def test_postprocess_normalized_coords_not_tiny_boxes() -> None:
    """NAKST ONNX exports xywh normalized to [0,1] — must not collapse to 1px boxes."""
    output = np.zeros((1, 17, 2), dtype=np.float32)
    # anchor 0: board-like full frame (should be filtered by skip_classes)
    output[0, 0, 0] = 0.5
    output[0, 1, 0] = 0.5
    output[0, 2, 0] = 1.0
    output[0, 3, 0] = 1.0
    output[0, 4, 0] = 0.99  # board class
    # anchor 1: pawn-sized box
    output[0, 0, 1] = 0.65
    output[0, 1, 1] = 0.70
    output[0, 2, 1] = 0.08
    output[0, 3, 1] = 0.10
    output[0, 4 + 12, 1] = 0.85  # black_pawn index 12

    class_names = (
        "board",
        "white_king", "white_queen", "white_rook", "white_bishop", "white_knight", "white_pawn",
        "black_king", "black_queen", "black_rook", "black_bishop", "black_knight", "black_pawn",
    )
    dets = _postprocess(
        output,
        orig_w=800,
        orig_h=800,
        scale=0.533,
        input_size=640,
        class_names=class_names,
        skip_classes=frozenset({"board"}),
        conf_threshold=0.25,
        iou_threshold=0.45,
        localization_only=False,
        board_class_index=0,
        coords_normalized=True,
        max_box_ratio=0.32,
        min_box_px=18,
    )
    assert len(dets) == 1
    assert dets[0].label == "black_pawn"
    _x, _y, w, h = dets[0].bbox
    assert w > 30 and h > 15
    assert w < 300 and h < 300


@pytest.mark.skipif(resolve_yolo_model(None) is None, reason="YOLO ONNX missing")
def test_yolo_on_rectified_board_finds_multiple_pieces() -> None:
    from vision.lc2fen.geometry import rectify_board
    from vision.inference.yolo_detector import get_yolo_piece_classifier

    test = Path(__file__).resolve().parents[3] / "ml" / "vendor" / "LiveChess2FEN" / "data" / "predictions" / "TestImages" / "FullDetection" / "test1.jpg"
    if not test.is_file():
        pytest.skip("test image missing")

    job = Path(__file__).resolve().parents[1] / "uploads" / "_test_yolo_post"
    job.mkdir(parents=True, exist_ok=True)
    (job / "input.jpg").write_bytes(test.read_bytes())
    board = rectify_board(job / "input.jpg").rectified_bgr
    dets = get_yolo_piece_classifier(0.25).detect(board)
    assert len(dets) >= 8
    for det in dets:
        _x, _y, w, h = det.bbox
        assert w > 20 and h > 20
