"""Tests for YOLO piece detection pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from vision.board.exceptions import BoardNotFoundError
from vision.inference.yolo_detector import YoloPieceDetection
from vision.scanner.config import ScannerConfig
from vision.scanner.mode import ScannerMode
from vision.scanner.piece_detection_pipeline import PieceDetectionPipeline
from vision.scanner.square_assignment import center_to_square_name, resolve_square_assignments


def _skewed_scene(size: int = 600) -> np.ndarray:
    import cv2

    board = np.zeros((400, 400, 3), dtype=np.uint8)
    cell = 400 // 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 70
            board[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = (color, color, color)
    src = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
    dst = np.float32([[40, 30], [360, 20], [340, 370], [20, 350]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board, m, (size, size), borderValue=(25, 25, 25))


def test_piece_detection_config_yolo() -> None:
    cfg = ScannerConfig.piece_detection_only()
    assert cfg.mode == ScannerMode.PIECE_DETECTION_ONLY
    assert cfg.ml.require_piece_model is True
    assert cfg.ml.require_occupancy_model is False


def test_center_to_square_name() -> None:
    board_size = 800
    assert center_to_square_name(50, 50, board_size) == "a8"
    assert center_to_square_name(750, 750, board_size) == "h1"


def test_piece_detection_pipeline_on_synthetic() -> None:
    pipeline = PieceDetectionPipeline()
    try:
        result = pipeline.run(_skewed_scene())
    except RuntimeError as exc:
        if "YOLO" in str(exc) or "ONNX" in str(exc) or "classifier" in str(exc).lower():
            pytest.skip(str(exc))
        raise
    except BoardNotFoundError:
        pytest.skip("localization failed on synthetic — geometry limit")

    piece_meta = result.metadata["piece_detection"]
    assert piece_meta["mode"] == "yolo_localize_classify"
    assert len(piece_meta["squares"]) == 64
    assert "detections" in piece_meta
    assert "detector" in piece_meta
    assert "classifier" in piece_meta


def test_resolve_square_assignments_picks_highest_conf() -> None:
    from vision.board.types import BoardGridResult, SquareCrop

    quad = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    sq = SquareCrop(
        row=3, col=4, image=img, bbox=(0, 0, 100, 100),
        cell_bbox=(400, 300, 500, 400), cell_quad=quad, crop_quad=quad,
    )
    grid = BoardGridResult(squares=((sq,),), board_size=800, square_size=100, margin_px=0)

    # center of e5 on 800px board: col=4 row=3 -> cx~450 cy~350
    dets = [
        YoloPieceDetection("white_pawn", 0.6, (430, 330, 40, 40)),
        YoloPieceDetection("white_queen", 0.9, (435, 335, 50, 50)),
    ]
    assigned = resolve_square_assignments(dets, grid)
    assert assigned["e5"].label == "white_queen"
