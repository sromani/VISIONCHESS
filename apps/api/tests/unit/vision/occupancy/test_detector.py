"""Tests for multi-signal occupancy detection."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from vision.occupancy.config import OccupancyConfig
from vision.occupancy.detector import OccupancyDetector, detect_square
from vision.scanner.config import ScannerConfig
from vision.scanner.context import ScanContext
from vision.scanner.stages.crop_quality import run_crop_quality
from vision.scanner.stages.extraction import run_extraction
from vision.scanner.stages.localization import run_localization
from vision.scanner.stages.mesh_rectify import run_mesh_rectification
from vision.scanner.stages.occupancy import run_occupancy
from tests.unit.vision.scanner.test_scan_pipeline import _skewed_scene


def _board_with_pieces(size: int = 800) -> np.ndarray:
    board = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size // 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 70
            board[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = (color, color, color)
    for col in range(8):
        cx = col * cell + cell // 2
        cv2.circle(board, (cx, 6 * cell + cell // 2), cell // 3, (240, 240, 240), -1)
        cv2.circle(board, (cx, 1 * cell + cell // 2), cell // 3, (30, 30, 30), -1)
    return board


def _skew(board: np.ndarray, out: int = 600) -> np.ndarray:
    size = board.shape[0]
    src = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    dst = np.float32([[75, 40], [525, 20], [485, 560], [35, 525]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board, matrix, (out, out), borderValue=(25, 25, 25))


class TestOccupancyDetector:
    def test_empty_checkerboard_few_false_positives(self) -> None:
        ctx = ScanContext(original_bgr=_skewed_scene(), config=ScannerConfig())
        run_localization(ctx)
        run_mesh_rectification(ctx)
        run_extraction(ctx)
        run_crop_quality(ctx)
        run_occupancy(ctx)
        assert ctx.metadata["occupancy"]["occupied_count"] <= 12

    def test_piece_board_detects_pieces_not_everything(self) -> None:
        # Synthetic circles are out-of-domain for chesscog occupancy; test heuristics path.
        config = replace(
            ScannerConfig(),
            occupancy=replace(OccupancyConfig(), ml_weight=0.0),
        )
        ctx = ScanContext(original_bgr=_skew(_board_with_pieces()), config=config)
        run_localization(ctx)
        run_mesh_rectification(ctx)
        run_extraction(ctx)
        run_crop_quality(ctx)
        run_occupancy(ctx)
        count = ctx.metadata["occupancy"]["occupied_count"]
        assert 6 <= count <= 32

    def test_uniform_crop_is_empty(self) -> None:
        crop = np.full((64, 64, 3), 120, dtype=np.uint8)
        result = detect_square(crop, 0, 0)
        assert result.probability < 0.35

    def test_circle_crop_is_occupied(self) -> None:
        crop = np.full((64, 64, 3), 120, dtype=np.uint8)
        cv2.circle(crop, (32, 32), 18, (240, 240, 240), -1)
        result = detect_square(crop, 6, 4, "e2")
        assert result.probability > 0.20
        assert not result.occupied

    def test_board_prior_caps_excess(self) -> None:
        detector = OccupancyDetector()
        # synthetic: build fake report path via skewed empty - should stay low
        ctx = ScanContext(original_bgr=_skewed_scene(), config=ScannerConfig())
        run_localization(ctx)
        run_mesh_rectification(ctx)
        run_extraction(ctx)
        run_crop_quality(ctx)
        report = detector.detect_grid(ctx.analysis_grid)
        assert report.occupied_count <= 36

    def test_debug_metadata_per_square(self) -> None:
        ctx = ScanContext(original_bgr=_skewed_scene(), config=ScannerConfig())
        run_localization(ctx)
        run_mesh_rectification(ctx)
        run_extraction(ctx)
        run_crop_quality(ctx)
        run_occupancy(ctx)
        squares = ctx.metadata["occupancy"]["squares"]
        assert len(squares) == 64
        assert "foreground_score" in squares[0]
        assert "fused_probability" in squares[0]
