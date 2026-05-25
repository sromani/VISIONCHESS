"""Run ScanPipeline on labeled real photos and compute metrics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from vision.benchmark.fen_grid import labels_to_placement, placement_to_labels
from vision.benchmark.manifest import BenchmarkCase, load_manifest
from vision.benchmark.metrics import BenchmarkAggregate, BenchmarkCaseResult, compare_grids
from vision.board.exceptions import BoardNotFoundError
from vision.classification.types import SquareClassification
from vision.scanner.config import ScannerConfig
from vision.scanner.pipeline import ScanPipeline


class RealPhotoBenchmarkRunner:
    def __init__(
        self,
        *,
        manifest_path: Path | None = None,
        config: ScannerConfig | None = None,
    ) -> None:
        self._manifest_path = manifest_path
        self._config = config or _benchmark_scanner_config()
        self._pipeline = ScanPipeline(self._config)

    def run_all(self, *, limit: int | None = None) -> BenchmarkAggregate:
        cases = load_manifest(self._manifest_path)
        if limit is not None:
            cases = cases[:limit]

        aggregate = BenchmarkAggregate()
        for case in cases:
            aggregate.add_case(self.run_case(case))
        return aggregate

    def run_case(self, case: BenchmarkCase) -> BenchmarkCaseResult:
        expected = placement_to_labels(case.placement)

        if not case.image.exists():
            return _failed_case(
                case,
                expected,
                error=f"Image not found: {case.image}",
            )

        bgr = cv2.imread(str(case.image))
        if bgr is None:
            return _failed_case(case, expected, error=f"Failed to read image: {case.image}")

        try:
            result = self._pipeline.run(bgr)
        except BoardNotFoundError as exc:
            return _failed_case(case, expected, error=f"localization_failed: {exc}")
        except Exception as exc:
            return _failed_case(case, expected, error=f"pipeline_error: {exc}")

        predicted_grid = _grid_from_squares(result.classification.squares)
        predicted_placement = labels_to_placement(predicted_grid)
        comparisons, metrics = compare_grids(expected, predicted_grid)

        timing = result.metadata.get("timing_ms", {})
        if not isinstance(timing, dict):
            timing = {}

        return BenchmarkCaseResult(
            case_id=case.id,
            image_path=str(case.image),
            expected_placement=case.placement,
            predicted_placement=predicted_placement,
            predicted_fen=result.classification.fen,
            fen_is_valid=result.classification.is_valid,
            pipeline_error=None,
            localization_ok=True,
            per_square=comparisons,
            piece_accuracy=metrics["piece_accuracy"],
            per_square_accuracy=metrics["per_square_accuracy"],
            occupancy_f1=metrics["occupancy_f1"],
            occupancy_precision=metrics["occupancy_precision"],
            occupancy_recall=metrics["occupancy_recall"],
            full_board_exact=predicted_placement == case.placement,
            timing_ms={k: int(v) for k, v in timing.items()},
        )


def _benchmark_scanner_config() -> ScannerConfig:
    base = ScannerConfig.production()
    return replace(
        base,
        collect_debug=False,
        use_stockfish_scoring=False,
        ml=replace(base.ml, use_stockfish_scoring=False),
    )


def _grid_from_squares(squares: tuple[SquareClassification, ...]) -> list[list[str]]:
    grid = [["empty"] * 8 for _ in range(8)]
    for sq in squares:
        grid[sq.row][sq.col] = sq.label if sq.occupied else "empty"
    return grid


def _failed_case(case: BenchmarkCase, expected: list[list[str]], *, error: str) -> BenchmarkCaseResult:
    empty_pred = [["empty"] * 8 for _ in range(8)]
    comparisons, metrics = compare_grids(expected, empty_pred)
    return BenchmarkCaseResult(
        case_id=case.id,
        image_path=str(case.image),
        expected_placement=case.placement,
        predicted_placement=labels_to_placement(empty_pred),
        predicted_fen="",
        fen_is_valid=False,
        pipeline_error=error,
        localization_ok=not error.startswith("localization"),
        per_square=comparisons,
        piece_accuracy=metrics["piece_accuracy"],
        per_square_accuracy=metrics["per_square_accuracy"],
        occupancy_f1=metrics["occupancy_f1"],
        occupancy_precision=metrics["occupancy_precision"],
        occupancy_recall=metrics["occupancy_recall"],
        full_board_exact=False,
        timing_ms={},
    )
