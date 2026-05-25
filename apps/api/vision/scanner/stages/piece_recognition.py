"""Soft-fusion piece recognition + context crop A/B experiment."""

from __future__ import annotations

from vision.classification.classifier import ClassifierConfig, create_classifier
from vision.classification.context_crop import build_context_grid, compare_crop_predictions
from vision.inference.ml_debug_types import PieceMlDebug
from vision.inference.model_registry import resolve_occupancy_model, resolve_piece_model
from vision.inference.piece_pipeline import PieceInferencePipeline
from vision.inference.runtime import InferenceRuntime
from vision.scanner.context import ScanContext
from vision.scanner.debug.context_viz import render_crop_comparison_montage
from vision.scanner.fusion import FusionConfig, compare_fusion_strategies, fuse_square
from vision.scanner.stages.ml_debug import run_ml_debug, store_piece_ml_debug_map
from vision.scanner.stages.occupancy import run_occupancy


def require_piece_onnx(ctx: ScanContext) -> PieceInferencePipeline:
    artifact = resolve_piece_model(ctx.config.classification.model_path)
    if artifact is None:
        runtime = InferenceRuntime.load()
        if not runtime.piece_available:
            msg = (
                "Piece classifier ONNX required but not found. "
                "Run: cd ml && python scripts/export_chesscog_standalone.py"
            )
            raise RuntimeError(msg)

    backend = create_classifier(
        ClassifierConfig(
            backend="ml",
            model_path=ctx.config.classification.model_path,
            allow_heuristic_fallback=False,
        ),
    )
    if not isinstance(backend, PieceInferencePipeline):
        msg = f"Expected ML ONNX backend, got {backend.name!r}. Heuristic fallback is disabled."
        raise RuntimeError(msg)
    return backend


def require_occupancy_onnx(ctx: ScanContext) -> None:
    artifact = resolve_occupancy_model(ctx.config.occupancy.ml_model_path)
    if artifact is None:
        runtime = InferenceRuntime.load()
        if not runtime.occupancy_available:
            msg = (
                "Occupancy ONNX required but not found. "
                "Run: cd ml && python scripts/export_chesscog_standalone.py"
            )
            raise RuntimeError(msg)


def _fusion_config(ctx: ScanContext) -> FusionConfig:
    occ = ctx.config.occupancy
    mode = occ.fusion_mode
    if mode not in {"soft_weighted", "product", "hard_gate"}:
        mode = "soft_weighted"
    return FusionConfig(
        mode=mode,  # type: ignore[arg-type]
        alpha=occ.fusion_alpha,
        beta=occ.fusion_beta,
        fused_empty_threshold=occ.fused_empty_threshold,
        hard_gate_threshold=occ.occupied_threshold,
    )


def _prediction_from_debug(dbg: PieceMlDebug) -> dict:
    top = dbg.top3[0] if dbg.top3 else None
    return {
        "label": top.label if top else "unknown",
        "confidence": top.probability if top else 0.0,
        "top3": [p.to_dict() for p in dbg.top3],
    }


def _run_context_crop_experiment(
    ctx: ScanContext,
    piece_pipeline: PieceInferencePipeline,
    tight_debug_list: list[PieceMlDebug],
    squares: list,
) -> tuple[dict[str, dict], dict | None]:
    ds_cfg = ctx.config.dataset_square
    if not ds_cfg.context_crop_enabled or ctx.rectified_board is None or ctx.analysis_grid is None:
        return {}, None

    scale = float(ds_cfg.context_crop_scale)
    ctx.context_grid = build_context_grid(
        ctx.rectified_board,
        ctx.analysis_grid,
        scale=scale,
        enhance_config=ds_cfg,
    )
    context_squares = list(ctx.context_grid.flat)
    _, context_debug_list = piece_pipeline.classify_squares_with_debug(context_squares)

    tight_by_name: dict[str, dict] = {}
    context_by_name: dict[str, dict] = {}
    montage_preds: dict[str, dict] = {}

    for sq, tight_dbg, ctx_dbg in zip(squares, tight_debug_list, context_debug_list, strict=True):
        tight_pred = _prediction_from_debug(tight_dbg)
        context_pred = _prediction_from_debug(ctx_dbg)
        ctx.context_piece_ml_debug_map[sq.square_name] = ctx_dbg
        tight_by_name[sq.square_name] = tight_pred
        context_by_name[sq.square_name] = context_pred
        montage_preds[sq.square_name] = {
            "tight_label": tight_pred["label"],
            "tight_confidence": tight_pred["confidence"],
            "context_label": context_pred["label"],
            "context_confidence": context_pred["confidence"],
        }

    experiment = compare_crop_predictions(tight_by_name, context_by_name)
    experiment["context_scale"] = scale

    if ctx.config.collect_debug:
        ctx.add_debug(
            "crop_comparison",
            render_crop_comparison_montage(
                ctx.analysis_grid,
                ctx.context_grid,
                predictions=montage_preds,
            ),
        )

    return context_by_name, experiment


def run_piece_recognition(ctx: ScanContext) -> None:
    if ctx.analysis_grid is None:
        msg = "Crop quality must run before piece recognition"
        raise ValueError(msg)

    require_occupancy_onnx(ctx)
    run_occupancy(ctx)

    fusion_cfg = _fusion_config(ctx)
    squares = list(ctx.analysis_grid.flat)

    piece_pipeline = require_piece_onnx(ctx)
    piece_artifact = piece_pipeline.artifact
    occ_artifact = resolve_occupancy_model(ctx.config.occupancy.ml_model_path)

    _, piece_debug_list = piece_pipeline.classify_squares_with_debug(squares)
    piece_debug_by_name: dict[str, PieceMlDebug] = {}
    raw_by_name: list[dict] = []

    for sq, dbg in zip(squares, piece_debug_list, strict=True):
        piece_debug_by_name[sq.square_name] = dbg
        top = dbg.top3[0] if dbg.top3 else None
        occ = ctx.occupancy[sq.square_name]
        ml_occ = ctx.occupancy_ml_debug_map.get(sq.square_name)
        occ_prob = ml_occ.occupied_probability if ml_occ is not None else occ.probability

        piece_label = top.label if top else "unknown"
        piece_conf = top.probability if top else 0.0
        raw_by_name.append(
            {
                "square_name": sq.square_name,
                "piece_label": piece_label,
                "piece_confidence": piece_conf,
                "occupancy_probability": occ_prob,
            }
        )

    store_piece_ml_debug_map(ctx, piece_debug_by_name)

    context_by_name, crop_experiment = _run_context_crop_experiment(
        ctx, piece_pipeline, piece_debug_list, squares
    )

    detections = []
    for sq, dbg, raw in zip(squares, piece_debug_list, raw_by_name, strict=True):
        fused = fuse_square(
            square_name=sq.square_name,
            piece_label=raw["piece_label"],
            piece_confidence=raw["piece_confidence"],
            occupancy_probability=raw["occupancy_probability"],
            cfg=fusion_cfg,
        )
        ctx_pred = context_by_name.get(sq.square_name, {})
        detections.append(
            {
                "square_name": sq.square_name,
                "row": sq.row,
                "col": sq.col,
                "occupied": fused.occupied,
                "occupancy_probability": fused.occupancy_probability,
                "empty_probability": fused.empty_probability,
                "piece_label": fused.piece_label,
                "piece_confidence": fused.piece_confidence,
                "fused_confidence": fused.fused_confidence,
                "fusion_mode": fused.fusion_mode,
                "hard_gate_occupied": fused.hard_gate_occupied,
                "hard_gate_label": fused.hard_gate_label,
                "occupancy_threshold": fusion_cfg.hard_gate_threshold,
                "fused_empty_threshold": fusion_cfg.fused_empty_threshold,
                "piece_classifier_ran": True,
                "label": fused.label,
                "confidence": fused.confidence,
                "top3": [p.to_dict() for p in dbg.top3],
                "logits": list(dbg.logits),
                "class_names": list(dbg.class_names),
                "cell_bbox": list(sq.cell_bbox),
                "context_piece_label": ctx_pred.get("label"),
                "context_piece_confidence": ctx_pred.get("confidence"),
                "context_top3": ctx_pred.get("top3", []),
            }
        )

    occupied_count = sum(1 for d in detections if d["occupied"])
    experiment = compare_fusion_strategies(raw_by_name, cfg=fusion_cfg)

    ctx.metadata["piece_detection"] = {
        "mode": "soft_fusion_piece_detection",
        "fusion": {
            "mode": fusion_cfg.mode,
            "alpha": fusion_cfg.alpha,
            "beta": fusion_cfg.beta,
            "fused_empty_threshold": fusion_cfg.fused_empty_threshold,
            "hard_gate_threshold": fusion_cfg.hard_gate_threshold,
        },
        "crop_experiment": crop_experiment,
        "occupied_count": occupied_count,
        "empty_count": 64 - occupied_count,
        "occupancy_model": _model_info(occ_artifact) if occ_artifact else {},
        "piece_model": {
            "source": piece_artifact.source,
            "path": str(piece_artifact.path),
            "architecture": piece_artifact.path.stem,
            "image_size": piece_artifact.image_size,
            "num_classes": piece_artifact.num_classes,
            "class_names": list(piece_artifact.class_names),
            "includes_empty_class": piece_artifact.includes_empty_class,
            "preprocess": "resize_rgb + imagenet_normalize (mean/std)",
        },
        "fusion_experiment": experiment,
        "squares": detections,
    }

    run_ml_debug(ctx)


def _model_info(artifact) -> dict:
    return {
        "source": artifact.source,
        "path": str(artifact.path),
        "image_size": artifact.image_size,
        "num_classes": artifact.num_classes,
        "preprocess": "resize_rgb + imagenet_normalize (mean/std)",
    }
