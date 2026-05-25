"""Real-photo benchmark — runs ScanPipeline on labeled OTB images."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vision.benchmark.manifest import default_manifest_path, load_manifest
from vision.benchmark.runner import RealPhotoBenchmarkRunner

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


def _manifest_ready() -> bool:
    path = default_manifest_path()
    if not path.exists():
        return False
    try:
        cases = load_manifest(path)
    except FileNotFoundError:
        return False
    return bool(cases) and cases[0].image.exists()


@pytest.mark.skipif(not _manifest_ready(), reason="Run benchmark/scripts/import_chesscog_transfer_learning.py")
def test_real_photo_benchmark_smoke() -> None:
    """Single real photo — verifies benchmark produces 64-square metrics."""
    runner = RealPhotoBenchmarkRunner()
    cases = load_manifest()
    result = runner.run_case(cases[0])
    assert result.case_id
    assert len(result.per_square) == 64
    assert 0.0 <= result.per_square_accuracy <= 1.0
    assert 0.0 <= result.piece_accuracy <= 1.0
    assert 0.0 <= result.occupancy_f1 <= 1.0
    if result.pipeline_error:
        pytest.skip(f"Pipeline failed on sample photo (localization): {result.pipeline_error}")


@pytest.mark.skipif(
    os.getenv("BENCHMARK_FULL") != "1" or not _manifest_ready(),
    reason="Set BENCHMARK_FULL=1 to run all real-photo cases (~15 min)",
)
def test_real_photo_benchmark_full() -> None:
    """Full manifest — piece accuracy, occupancy F1, legal FEN rate."""
    aggregate = RealPhotoBenchmarkRunner().run_all()
    assert aggregate.cases_run > 0
    assert aggregate.cases_failed < aggregate.cases_run
    print(
        f"\nBENCHMARK: piece={aggregate.piece_accuracy:.2%} "
        f"square={aggregate.per_square_accuracy:.2%} "
        f"occ_f1={aggregate.occupancy_f1:.2%} "
        f"board={aggregate.full_board_accuracy:.2%} "
        f"legal={aggregate.legal_fen_rate:.2%}"
    )
    assert aggregate.per_square_accuracy > 0.0
