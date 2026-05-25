"""Phase 5 — ML piece classification (batched ONNX, no heuristics)."""

from __future__ import annotations

from dataclasses import replace

from vision.classification.classifier import ClassifierConfig, create_classifier
from vision.classification.types import SquareClassification
from vision.inference.piece_pipeline import PieceInferencePipeline
from vision.scanner.context import ScanContext


def run_classification(ctx: ScanContext) -> list[SquareClassification]:
    if ctx.analysis_grid is None:
        msg = "Crop quality must run before classification"
        raise ValueError(msg)

    grid = ctx.analysis_grid
    ml_cfg = ctx.config.ml

    piece_backend = create_classifier(
        ClassifierConfig(
            backend="ml",
            model_path=ctx.config.classification.model_path,
            allow_heuristic_fallback=ml_cfg.allow_heuristic_fallback,
            use_tta=False,
        ),
    )

    preds: list[SquareClassification]
    if isinstance(piece_backend, PieceInferencePipeline) and ctx.config.ml.capture_ml_debug:
        preds, piece_debug = piece_backend.classify_squares_with_debug(list(grid.flat))
        from vision.scanner.stages.ml_debug import store_piece_ml_debug

        store_piece_ml_debug(ctx, piece_debug)
    else:
        preds = piece_backend.classify_squares(list(grid.flat), soft=True)
    results: list[SquareClassification] = []
    for sq in preds:
        occ_prob = _occ_prob(ctx, sq.square_name)
        results.append(
            replace(
                sq,
                occupied=False,
                occupancy_score=occ_prob,
                empty_reason=None,
            )
        )

    ctx.squares = results
    ctx.metadata["classification"] = {
        "backend": piece_backend.name,
        "mode": "ml_soft",
        "input_grid": "analysis_high_res",
        "input_square_px": grid.square_size,
        "squares_classified": len(results),
        "heuristics": piece_backend.name == "heuristic",
        "batched_inference": isinstance(piece_backend, PieceInferencePipeline),
    }
    return results


def _occ_prob(ctx: ScanContext, square_name: str) -> float:
    occ = ctx.occupancy.get(square_name)
    return occ.probability if occ is not None else 0.5
