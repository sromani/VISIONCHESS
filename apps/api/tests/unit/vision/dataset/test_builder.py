"""Tests for dataset builder layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision.classification.labels import CLASS_NAMES
from vision.dataset.builder import DatasetBuilder


class TestDatasetBuilder:
    def test_save_square_creates_class_folder(self, tmp_path: Path) -> None:
        builder = DatasetBuilder(tmp_path, split="train")
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        sample = builder.save_square(
            img,
            "white_pawn",
            square_name="e2",
            confidence=0.91,
            job_id="job1",
            orientation="normal",
        )
        assert sample.path.exists()
        assert sample.path.parent.name == "white_pawn"
        assert (tmp_path / "train" / "empty").is_dir()
        assert (tmp_path / "manifest.jsonl").exists()

    def test_relabel_moves_file(self, tmp_path: Path) -> None:
        builder = DatasetBuilder(tmp_path, split="train")
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        sample = builder.save_square(img, "empty", square_name="a1", confidence=0.8)
        dest = builder.relabel(sample.path, "black_rook")
        assert dest.exists()
        assert dest.parent.name == "black_rook"
        assert not sample.path.exists()
        assert (tmp_path / "corrections.jsonl").exists()

    def test_all_class_dirs_exist(self, tmp_path: Path) -> None:
        DatasetBuilder(tmp_path)
        for name in CLASS_NAMES:
            assert (tmp_path / "train" / name).is_dir()
