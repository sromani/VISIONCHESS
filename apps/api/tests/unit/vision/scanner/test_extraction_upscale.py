"""Tests for rectified board upscale before extraction."""

from __future__ import annotations

import cv2
import numpy as np

from vision.board.upscale import upscale_rectified_board
from vision.scanner.config import ScannerConfig
from vision.scanner.context import ScanContext
from vision.scanner.stages.crop_quality import run_crop_quality
from vision.scanner.stages.extraction import run_extraction
from vision.scanner.stages.localization import run_localization
from vision.scanner.stages.mesh_rectify import run_mesh_rectification
from tests.unit.vision.scanner.test_scan_pipeline import _skewed_scene


class TestExtractionUpscale:
    def test_upscale_rectified_board_to_target(self) -> None:
        board = np.zeros((800, 800, 3), dtype=np.uint8)
        out = upscale_rectified_board(board, 2048)
        assert out.shape == (2048, 2048, 3)

    def test_pipeline_produces_high_res_raw_crops(self) -> None:
        ctx = ScanContext(original_bgr=_skewed_scene(), config=ScannerConfig())
        run_localization(ctx)
        run_mesh_rectification(ctx)
        run_extraction(ctx)
        run_crop_quality(ctx)

        assert ctx.rectified_board is not None
        assert ctx.rectified_board.shape[:2] == (2048, 2048)
        assert ctx.raw_grid is not None
        assert ctx.raw_grid.square_size >= 180
        assert ctx.analysis_grid is not None
        assert ctx.analysis_grid.square_size >= 128
        assert ctx.dataset_grid is None
        assert ctx.metadata["extraction"]["raw_square_px"] >= 180
        assert ctx.metadata["crop_quality"]["analysis_square_px"] >= 128

    def test_upscale_disabled_keeps_mesh_size(self) -> None:
        from dataclasses import replace
        from vision.board.grid_config import GridExtractorConfig

        cfg = replace(ScannerConfig().grid, upscale_enabled=False)
        ctx = ScanContext(
            original_bgr=_skewed_scene(),
            config=replace(ScannerConfig(), grid=cfg),
        )
        run_localization(ctx)
        run_mesh_rectification(ctx)
        run_extraction(ctx)

        assert ctx.rectified_board is not None
        assert ctx.rectified_board.shape[0] == 800
