"""Two-stage piece detection: YOLO localization + classifier on crops."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import cv2
import numpy as np

from vision.classification.piece_crop import extract_piece_crop, resize_crop_preview
from vision.inference.piece_pipeline import PieceInferenceConfig, PieceInferencePipeline
from vision.inference.yolo_detector import YoloPieceDetection, get_yolo_detector
from vision.scanner.context import ScanContext
from vision.scanner.debug.yolo_viz import render_classification_overlay, render_crop_montage, render_localization_overlay
from vision.scanner.square_assignment import build_classified_square_records, resolve_square_assignments


@dataclass(slots=True)
class ClassifiedPiece:
    localization: YoloPieceDetection
    piece_label: str
    piece_confidence: float
    top3: list[dict]
    crop_shape: tuple[int, int]


def run_yolo_detection(ctx: ScanContext) -> None:
    """Stage A: YOLO localizes pieces. Stage B: classifier labels each crop."""
    if ctx.rectified_board is None:
        raise ValueError("rectified board must exist before YOLO detection")
    if ctx.raw_grid is None:
        raise ValueError("grid extraction must run before YOLO detection")

    board = ctx.rectified_board
    detector = get_yolo_detector(None)

    t0 = time.perf_counter()
    localizations = detector.detect_localization(board)
    assigned_loc = resolve_square_assignments(localizations, ctx.raw_grid)
    loc_ms = int((time.perf_counter() - t0) * 1000)

    # Stage B — classify one crop per occupied square
    classifier = _require_classifier(ctx)
    ordered_squares = sorted(assigned_loc.keys())
    crops = [extract_piece_crop(board, assigned_loc[sq].bbox) for sq in ordered_squares]

    t1 = time.perf_counter()
    classified_rows, debug_rows = classifier.classify_crops_with_debug(crops)
    cls_ms = int((time.perf_counter() - t1) * 1000)

    classified: list[ClassifiedPiece] = []
    detection_dicts: list[dict] = []
    for square_name, (label, conf, top3), loc in zip(
        ordered_squares, classified_rows, [assigned_loc[s] for s in ordered_squares], strict=True
    ):
        crop = extract_piece_crop(board, loc.bbox)
        top3_dicts = [
            {"label": t.label, "probability": t.probability, "logit": t.logit} for t in top3
        ]
        classified.append(
            ClassifiedPiece(
                localization=loc,
                piece_label=label,
                piece_confidence=conf,
                top3=top3_dicts,
                crop_shape=(crop.shape[0], crop.shape[1]),
            )
        )
        detection_dicts.append(
            {
                "class": "piece",
                "localization_confidence": loc.confidence,
                "classified_label": label,
                "classified_confidence": conf,
                "bbox": list(loc.bbox),
                "square": square_name,
                "top3": top3_dicts,
                "crop_shape": [crop.shape[0], crop.shape[1]],
            }
        )

    classified_by_square = {d["square"]: d for d in detection_dicts if d.get("square")}
    squares = build_classified_square_records(ctx.raw_grid, classified_by_square)
    occupied_count = sum(1 for s in squares if s["occupied"])

    ctx.metadata["piece_detection"] = {
        "mode": "yolo_localize_classify",
        "detector": detector.info_dict(),
        "classifier": _classifier_info(classifier),
        "detections": detection_dicts,
        "squares": squares,
        "occupied_count": occupied_count,
        "empty_count": 64 - occupied_count,
        "timing_ms": {
            "yolo_localization_ms": loc_ms,
            "classifier_ms": cls_ms,
            "total_ml_ms": loc_ms + cls_ms,
        },
    }

    if ctx.config.collect_debug:
        loc_list = list(assigned_loc.values())
        ctx.add_debug("yolo_localization", render_localization_overlay(board, loc_list))
        ctx.add_debug("yolo_overlay", render_localization_overlay(board, loc_list))
        ctx.add_debug(
            "classifier_overlay",
            render_classification_overlay(board, classified_by_square),
        )
        ctx.add_debug(
            "piece_overlay",
            render_classification_overlay(board, classified_by_square),
        )
        ctx.add_debug(
            "classifier_crops",
            render_crop_montage(ordered_squares, crops, classified_by_square),
        )
        if debug_rows:
            ctx.metadata["_classifier_crop_debug"] = [
                {
                    "square": sq,
                    "top3": detection_dicts[i]["top3"],
                    "crop_base64": _encode_crop_jpeg(crops[i]),
                }
                for i, sq in enumerate(ordered_squares)
            ]


def _require_classifier(ctx: ScanContext) -> PieceInferencePipeline:
    path = ctx.config.classification.model_path
    return PieceInferencePipeline.from_registry(PieceInferenceConfig(model_path=path))


def _classifier_info(pipeline: PieceInferencePipeline) -> dict:
    art = pipeline.artifact
    return {
        "source": art.source,
        "path": str(art.path),
        "image_size": art.image_size,
        "num_classes": art.num_classes,
        "class_names": list(art.class_names),
        "preprocess": "imagenet_square",
        "model_type": "piece_classifier",
        "role": "fine_classifier",
    }


def _encode_crop_jpeg(crop: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
