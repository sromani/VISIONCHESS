"""Piece-detection-only API schemas — YOLO localize + classifier."""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel, Field

from app.api.schemas.common import PointSchema


class TopPredictionSchema(BaseModel):
    label: str
    probability: float
    logit: float


class PieceBBoxDetectionSchema(BaseModel):
    class_name: str = Field(alias="class")
    localization_confidence: float
    classified_label: str | None = None
    classified_confidence: float | None = None
    bbox: list[int]
    square: str | None = None
    top3: list[TopPredictionSchema] = Field(default_factory=list)
    crop_shape: list[int] | None = None

    model_config = {"populate_by_name": True}


class SquareAssignmentSchema(BaseModel):
    square_name: str
    row: int
    col: int
    occupied: bool
    label: str
    confidence: float
    piece_label: str
    piece_confidence: float
    localization_confidence: float = 0.0
    bbox: list[int] | None = None
    top3: list[TopPredictionSchema] = Field(default_factory=list)
    cell_bbox: list[int]


class DetectorModelInfoSchema(BaseModel):
    source: str
    path: str
    image_size: int
    num_classes: int
    class_names: list[str]
    preprocess: str
    model_type: str = "yolo_detector"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    localization_only: bool = True
    role: str = "piece_localizer"


class ClassifierModelInfoSchema(BaseModel):
    source: str
    path: str
    image_size: int
    num_classes: int
    class_names: list[str]
    preprocess: str
    model_type: str = "piece_classifier"
    role: str = "fine_classifier"


class ClassifierCropDebugSchema(BaseModel):
    square: str
    top3: list[TopPredictionSchema]
    crop_base64: str


class PieceDetectionDebugSchema(BaseModel):
    original_base64: str | None = None
    rectified_upscaled_base64: str | None = None
    yolo_localization_base64: str | None = None
    yolo_overlay_base64: str | None = None
    classifier_overlay_base64: str | None = None
    piece_overlay_base64: str | None = None
    classifier_crops_base64: str | None = None


class PieceDetectionResponse(BaseModel):
    job_id: str
    mode: str = "yolo_localize_classify"
    corners: list[PointSchema]
    homography: list[list[float]]
    confidence: float
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    rectified_board_base64: str
    detector: DetectorModelInfoSchema
    classifier: ClassifierModelInfoSchema
    detections: list[PieceBBoxDetectionSchema]
    squares: list[SquareAssignmentSchema]
    classifier_crops: list[ClassifierCropDebugSchema] = Field(default_factory=list)
    occupied_count: int = 0
    empty_count: int = 64
    debug: PieceDetectionDebugSchema | None = None
    processing_ms: int
    metadata: dict[str, Any]

    @classmethod
    def from_result(
        cls,
        *,
        job_id: str,
        result,
        debug_jpegs: dict[str, bytes],
        processing_ms: int,
    ) -> PieceDetectionResponse:
        piece_meta = result.metadata.get("piece_detection", {})
        detector_raw = piece_meta.get("detector", {})
        classifier_raw = piece_meta.get("classifier", {})
        detections_raw = piece_meta.get("detections", [])
        squares_raw = piece_meta.get("squares", [])
        crop_debug = result.metadata.get("_classifier_crop_debug", [])

        return cls(
            job_id=job_id,
            corners=[PointSchema(x=x, y=y) for x, y in result.corners_list],
            homography=result.homography_list(),
            confidence=result.confidence,
            original_width=result.original_width,
            original_height=result.original_height,
            output_width=result.output_width,
            output_height=result.output_height,
            rectified_board_base64=_encode(_jpeg(result.rectified_board)),
            detector=DetectorModelInfoSchema(
                source=str(detector_raw.get("source", "unknown")),
                path=str(detector_raw.get("path", "")),
                image_size=int(detector_raw.get("image_size", 640)),
                num_classes=int(detector_raw.get("num_classes", 13)),
                class_names=list(detector_raw.get("class_names", [])),
                preprocess=str(detector_raw.get("preprocess", "yolo_rgb_640")),
                model_type=str(detector_raw.get("model_type", "yolo_detector")),
                conf_threshold=float(detector_raw.get("conf_threshold", 0.25)),
                iou_threshold=float(detector_raw.get("iou_threshold", 0.45)),
                localization_only=bool(detector_raw.get("localization_only", True)),
                role=str(detector_raw.get("role", "piece_localizer")),
            ),
            classifier=ClassifierModelInfoSchema(
                source=str(classifier_raw.get("source", "unknown")),
                path=str(classifier_raw.get("path", "")),
                image_size=int(classifier_raw.get("image_size", 299)),
                num_classes=int(classifier_raw.get("num_classes", 12)),
                class_names=list(classifier_raw.get("class_names", [])),
                preprocess=str(classifier_raw.get("preprocess", "imagenet_square")),
                model_type=str(classifier_raw.get("model_type", "piece_classifier")),
                role=str(classifier_raw.get("role", "fine_classifier")),
            ),
            detections=[
                PieceBBoxDetectionSchema(
                    class_name=str(d.get("class", "piece")),
                    localization_confidence=float(d.get("localization_confidence", d.get("confidence", 0.0))),
                    classified_label=d.get("classified_label"),
                    classified_confidence=(
                        float(d["classified_confidence"]) if d.get("classified_confidence") is not None else None
                    ),
                    bbox=[int(x) for x in d["bbox"]],
                    square=d.get("square"),
                    top3=[TopPredictionSchema(**t) for t in d.get("top3", [])],
                    crop_shape=d.get("crop_shape"),
                )
                for d in detections_raw
            ],
            squares=[
                SquareAssignmentSchema(
                    square_name=str(s["square_name"]),
                    row=int(s["row"]),
                    col=int(s["col"]),
                    occupied=bool(s.get("occupied", False)),
                    label=str(s["label"]),
                    confidence=float(s.get("confidence", 0.0)),
                    piece_label=str(s.get("piece_label", s["label"])),
                    piece_confidence=float(s.get("piece_confidence", s.get("confidence", 0.0))),
                    localization_confidence=float(s.get("localization_confidence", 0.0)),
                    bbox=[int(x) for x in s["bbox"]] if s.get("bbox") else None,
                    top3=[TopPredictionSchema(**t) for t in s.get("top3", [])],
                    cell_bbox=[int(x) for x in s.get("cell_bbox", [])],
                )
                for s in squares_raw
            ],
            classifier_crops=[
                ClassifierCropDebugSchema(
                    square=str(c["square"]),
                    top3=[TopPredictionSchema(**t) for t in c.get("top3", [])],
                    crop_base64=str(c.get("crop_base64", "")),
                )
                for c in crop_debug
            ],
            occupied_count=int(piece_meta.get("occupied_count", 0)),
            empty_count=int(piece_meta.get("empty_count", 64)),
            debug=_build_debug(debug_jpegs),
            processing_ms=processing_ms,
            metadata=_public_metadata(result.metadata),
        )


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if not key.startswith("_")}


def _build_debug(debug_jpegs: dict[str, bytes]) -> PieceDetectionDebugSchema | None:
    if not debug_jpegs:
        return None
    return PieceDetectionDebugSchema(
        original_base64=_encode(debug_jpegs.get("original")),
        rectified_upscaled_base64=_encode(debug_jpegs.get("rectified_upscaled")),
        yolo_localization_base64=_encode(debug_jpegs.get("yolo_localization")),
        yolo_overlay_base64=_encode(debug_jpegs.get("yolo_overlay")),
        classifier_overlay_base64=_encode(debug_jpegs.get("classifier_overlay")),
        piece_overlay_base64=_encode(debug_jpegs.get("piece_overlay")),
        classifier_crops_base64=_encode(debug_jpegs.get("classifier_crops")),
    )


def _encode(data: bytes | None) -> str | None:
    if data is None:
        return None
    return base64.b64encode(data).decode("ascii")


def _jpeg(image) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()
