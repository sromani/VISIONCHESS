"""YOLOv8 chess piece detector — full-board bounding boxes via ONNX."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from vision.inference.model_registry import resolve_yolo_model


@dataclass(frozen=True, slots=True)
class YoloPieceDetection:
    """Single piece detection on the rectified board."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    square_name: str | None = None


@dataclass(frozen=True, slots=True)
class YoloDetectorConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    input_size: int = 640
    skip_classes: frozenset[str] = frozenset({"board"})
    localization_only: bool = True
    board_class_index: int = 0
    coords_normalized: bool = False
    max_box_ratio: float = 0.35
    min_box_px: int = 16


class YoloPieceDetector:
    """Run YOLOv8 ONNX on a rectified board image."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        config: YoloDetectorConfig | None = None,
    ) -> None:
        artifact = resolve_yolo_model(model_path)
        if artifact is None:
            msg = (
                "YOLO chess detector ONNX not found. Run: "
                "python ml/scripts/setup_yolo_chess.py"
            )
            raise RuntimeError(msg)

        self._artifact = artifact
        self._config = config or YoloDetectorConfig(
            conf_threshold=float(artifact.extra.get("conf_threshold", 0.25)),
            iou_threshold=float(artifact.extra.get("iou_threshold", 0.45)),
            input_size=artifact.image_size,
            skip_classes=frozenset(artifact.extra.get("skip_classes", ["board"])),
            localization_only=bool(artifact.extra.get("localization_only", True)),
            board_class_index=int(artifact.extra.get("board_class_index", 0)),
            coords_normalized=bool(artifact.extra.get("coords_normalized", False)),
            max_box_ratio=float(artifact.extra.get("max_box_ratio", 0.35)),
            min_box_px=int(artifact.extra.get("min_box_px", 16)),
        )
        self._session = ort.InferenceSession(
            str(artifact.path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = artifact.input_name
        self._class_names = artifact.class_names

    @property
    def artifact(self):
        return self._artifact

    def detect(self, board_bgr: NDArray[np.uint8]) -> list[YoloPieceDetection]:
        """Detect pieces. In localization_only mode every box is labeled ``piece``."""
        h, w = board_bgr.shape[:2]
        blob, scale = _preprocess(board_bgr, self._config.input_size)
        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0]
        return _postprocess(
            raw,
            orig_w=w,
            orig_h=h,
            scale=scale,
            input_size=self._config.input_size,
            class_names=self._class_names,
            skip_classes=self._config.skip_classes,
            conf_threshold=self._config.conf_threshold,
            iou_threshold=self._config.iou_threshold,
            localization_only=self._config.localization_only,
            board_class_index=self._config.board_class_index,
            coords_normalized=self._config.coords_normalized,
            max_box_ratio=self._config.max_box_ratio,
            min_box_px=self._config.min_box_px,
        )

    def detect_localization(self, board_bgr: NDArray[np.uint8]) -> list[YoloPieceDetection]:
        """Piece-localization-only — bbox + confidence, no fine class."""
        return self.detect(board_bgr)

    def info_dict(self) -> dict[str, Any]:
        return {
            "source": self._artifact.source,
            "path": str(self._artifact.path),
            "image_size": self._artifact.image_size,
            "num_classes": self._artifact.num_classes,
            "class_names": list(self._class_names),
            "preprocess": "yolo_rgb_640",
            "model_type": "yolo_detector",
            "conf_threshold": self._config.conf_threshold,
            "iou_threshold": self._config.iou_threshold,
            "localization_only": self._config.localization_only,
            "coords_normalized": self._config.coords_normalized,
            "max_box_ratio": self._config.max_box_ratio,
            "role": "piece_localizer",
        }


@lru_cache(maxsize=1)
def get_yolo_detector(model_path: str | None = None) -> YoloPieceDetector:
    path = Path(model_path) if model_path else None
    return YoloPieceDetector(path)


@lru_cache(maxsize=4)
def get_yolo_piece_classifier(conf_threshold: float = 0.30) -> YoloPieceDetector:
    """YOLO with fine piece classes (not localization-only). Higher default conf for precision."""
    artifact = resolve_yolo_model(None)
    if artifact is None:
        msg = (
            "YOLO chess detector ONNX not found. Run: "
            "python ml/scripts/setup_yolo_chess.py"
        )
        raise RuntimeError(msg)
    return YoloPieceDetector(
        config=YoloDetectorConfig(
            conf_threshold=conf_threshold,
            iou_threshold=float(artifact.extra.get("iou_threshold", 0.45)),
            input_size=artifact.image_size,
            skip_classes=frozenset(artifact.extra.get("skip_classes", ["board"])),
            localization_only=False,
            board_class_index=int(artifact.extra.get("board_class_index", 0)),
            coords_normalized=bool(artifact.extra.get("coords_normalized", False)),
            max_box_ratio=float(artifact.extra.get("max_box_ratio", 0.35)),
            min_box_px=int(artifact.extra.get("min_box_px", 16)),
        ),
    )


def _preprocess(
    board_bgr: NDArray[np.uint8],
    input_size: int,
) -> tuple[NDArray[np.float32], float]:
    rgb = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = input_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return blob, scale


def _postprocess(
    output: NDArray[np.float32],
    *,
    orig_w: int,
    orig_h: int,
    scale: float,
    input_size: int,
    class_names: tuple[str, ...],
    skip_classes: frozenset[str],
    conf_threshold: float,
    iou_threshold: float,
    localization_only: bool = True,
    board_class_index: int = 0,
    coords_normalized: bool = False,
    max_box_ratio: float = 0.35,
    min_box_px: int = 16,
) -> list[YoloPieceDetection]:
    """Decode YOLOv8/v11 ONNX output → detections in original image coords."""
    pred = _reshape_predictions(output)
    if pred is None or pred.shape[1] < 5:
        return []

    boxes_xywh = pred[:, :4].astype(np.float32)
    class_scores = pred[:, 4:]

    if coords_normalized or float(boxes_xywh.max(initial=0.0)) <= 1.5:
        boxes_xywh *= float(input_size)

    if localization_only and class_scores.shape[1] > 1 and board_class_index >= 0:
        piece_scores = np.delete(class_scores, board_class_index, axis=1)
        confidences = piece_scores.max(axis=1)
        class_ids = piece_scores.argmax(axis=1)
        # Map back to full label index (after board removal)
        class_ids = np.where(class_ids >= board_class_index, class_ids + 1, class_ids)
    else:
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

    labels = np.array(
        [class_names[int(cid)] if int(cid) < len(class_names) else f"class_{int(cid)}" for cid in class_ids]
    )
    skip_mask = np.array([label not in skip_classes for label in labels])
    conf_mask = confidences >= conf_threshold
    mask = skip_mask & conf_mask

    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    labels = labels[mask]

    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    boxes_xyxy /= scale
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    max_w = orig_w * max_box_ratio
    max_h = orig_h * max_box_ratio
    size_mask = []
    for x1, y1, x2, y2 in boxes_xyxy:
        bw, bh = x2 - x1, y2 - y1
        ok = bw >= min_box_px and bh >= min_box_px and bw <= max_w and bh <= max_h
        size_mask.append(ok)
    size_mask_arr = np.array(size_mask, dtype=bool)
    boxes_xyxy = boxes_xyxy[size_mask_arr]
    confidences = confidences[size_mask_arr]
    class_ids = class_ids[size_mask_arr]
    labels = labels[size_mask_arr]

    if len(boxes_xyxy) == 0:
        return []

    keep = _nms(boxes_xyxy, confidences, iou_threshold)
    detections: list[YoloPieceDetection] = []

    for idx in keep:
        label = "piece" if localization_only else str(labels[idx])
        x1, y1, x2, y2 = boxes_xyxy[idx]
        detections.append(
            YoloPieceDetection(
                label=label,
                confidence=float(confidences[idx]),
                bbox=(int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))),
            )
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _reshape_predictions(output: NDArray[np.float32]) -> NDArray[np.float32] | None:
    if output.ndim != 3:
        return None
    batch = output[0]
    if batch.ndim != 2:
        return None
    rows, cols = batch.shape
    # Ultralytics ONNX layout is [4+classes, num_anchors]; convert to [N, 4+classes].
    if rows <= 64 and cols > rows:
        return batch.T
    if cols <= 64 and rows > cols:
        return batch.T
    return batch


def _xywh_to_xyxy(boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], iou_threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
