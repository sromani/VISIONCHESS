"""Tests for crop quality stage — high-res before ML."""

from __future__ import annotations

import numpy as np

from vision.classification.square_quality import enhance_square_crop
from vision.scanner.config import ScannerConfig
from vision.scanner.context import ScanContext
from vision.scanner.stages.crop_quality import run_crop_quality
from vision.scanner.stages.extraction import run_extraction
from vision.scanner.stages.localization import run_localization
from vision.scanner.stages.mesh_rectify import run_mesh_rectification
from tests.unit.vision.scanner.test_scan_pipeline import _skewed_scene


class TestCropQuality:
    def test_enhance_preserves_resolution(self) -> None:
        crop = np.random.randint(0, 255, (216, 216, 3), dtype=np.uint8)
        out = enhance_square_crop(crop)
        assert out.shape[0] >= 128
        assert out.shape[0] <= 216

    def test_analysis_grid_before_dataset_downscale(self) -> None:
        ctx = ScanContext(original_bgr=_skewed_scene(), config=ScannerConfig())
        run_localization(ctx)
        run_mesh_rectification(ctx)
        run_extraction(ctx)
        run_crop_quality(ctx)

        assert ctx.analysis_grid is not None
        assert ctx.analysis_grid.square_size >= 128
        assert ctx.analysis_grid.square_size > 64
        assert ctx.dataset_grid is None

    def test_classification_uses_analysis_not_64px(self) -> None:
        ctx = ScanContext(original_bgr=_skewed_scene(), config=ScannerConfig())
        run_localization(ctx)
        run_mesh_rectification(ctx)
        run_extraction(ctx)
        run_crop_quality(ctx)

        first = ctx.analysis_grid.flat[0]
        assert first.image.shape[0] >= 128
