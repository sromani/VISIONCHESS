"""YOLO piece recognition on LC2FEN-rectified boards."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.inference.yolo_detector import YoloDetectorConfig, YoloPieceDetection, get_yolo_piece_classifier
from vision.lc2fen.common import SquarePrediction
from vision.lc2fen.fen_validate import FenValidation, validate_placement_fen
from vision.lc2fen.geometry import LC2FENGeometryResult, rectify_board_from_bytes
from vision.scanner.debug.yolo_viz import render_yolo_class_overlay
from vision.scanner.square_assignment import assign_squares, bbox_center

YOLO_TO_FEN: dict[str, str] = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
}


@dataclass
class YoloDetectionRecord:
    label: str
    confidence: float
    square: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


@dataclass
class LC2FENYoloResult:
    fen: str
    geometry: LC2FENGeometryResult
    square_predictions: list[SquarePrediction]
    raw_detections: list[YoloPieceDetection]
    assigned: dict[str, YoloPieceDetection]
    validation: FenValidation
    yolo_overlay_bgr: NDArray[np.uint8]
    metadata: dict = field(default_factory=dict)


def recognize_pieces_yolo(
    rectified_bgr: NDArray[np.uint8],
    *,
    conf_threshold: float = 0.35,
) -> tuple[
    list[YoloPieceDetection],
    dict[str, YoloPieceDetection],
    list[SquarePrediction],
    str,
    NDArray[np.uint8],
]:
    """Run YOLO object detection on rectified board and build sparse FEN."""
    board_size = min(rectified_bgr.shape[0], rectified_bgr.shape[1])
    detector = get_yolo_piece_classifier(conf_threshold=conf_threshold)
    raw = detector.detect(rectified_bgr)
    assigned = _resolve_assignments(raw, board_size)
    square_predictions = _build_square_predictions(board_size, assigned)
    fen = _fen_from_assignments(assigned)
    overlay = render_yolo_class_overlay(rectified_bgr, list(assigned.values()))
    return raw, assigned, square_predictions, fen, overlay


def run_lc2fen_yolo_pipeline(
    geometry: LC2FENGeometryResult,
    *,
    conf_threshold: float = 0.35,
) -> LC2FENYoloResult:
    raw, assigned, square_predictions, fen, overlay = recognize_pieces_yolo(
        geometry.rectified_bgr,
        conf_threshold=conf_threshold,
    )
    validation = validate_placement_fen(fen)
    detector = get_yolo_piece_classifier(conf_threshold=conf_threshold)

    records = [
        YoloDetectionRecord(
            label=det.label,
            confidence=det.confidence,
            square=det.square_name or "?",
            bbox=det.bbox,
            center=bbox_center(det.bbox),
        )
        for det in raw
        if det.square_name
    ]

    return LC2FENYoloResult(
        fen=fen,
        geometry=geometry,
        square_predictions=square_predictions,
        raw_detections=raw,
        assigned=assigned,
        validation=validation,
        yolo_overlay_bgr=overlay,
        metadata={
            "backend": "lc2fen_geometry_yolo_pieces",
            "piece_detector": detector.info_dict(),
            "conf_threshold": conf_threshold,
            "detection_count": len(raw),
            "assigned_count": len(assigned),
            "yolo_detections": [
                {
                    "label": r.label,
                    "confidence": round(r.confidence, 4),
                    "square": r.square,
                    "bbox": list(r.bbox),
                    "center": [round(r.center[0], 1), round(r.center[1], 1)],
                }
                for r in records
            ],
        },
    )


def _resolve_assignments(
    detections: list[YoloPieceDetection],
    board_size: int,
) -> dict[str, YoloPieceDetection]:
    """One detection per square — highest confidence wins."""
    with_squares = assign_squares(detections, board_size)
    best: dict[str, YoloPieceDetection] = {}
    for det in with_squares:
        if not det.square_name:
            continue
        prev = best.get(det.square_name)
        if prev is None or det.confidence > prev.confidence:
            best[det.square_name] = det
    return best


def _build_square_predictions(
    board_size: int,
    assigned: dict[str, YoloPieceDetection],
) -> list[SquarePrediction]:
    cell = board_size // 8
    predictions: list[SquarePrediction] = []
    for row in range(8):
        for col in range(8):
            rank = 8 - row
            name = f"{chr(ord('a') + col)}{rank}"
            det = assigned.get(name)
            label = det.label if det else "empty"
            conf = float(det.confidence) if det else 0.0
            predictions.append(
                SquarePrediction(
                    name=name,
                    row=row,
                    col=col,
                    label=label,
                    confidence=conf,
                    occupied=det is not None,
                )
            )
    return predictions


def _fen_from_assignments(assigned: dict[str, YoloPieceDetection]) -> str:
    """Build piece-placement FEN from sparse YOLO detections."""
    by_square = assigned
    ranks: list[str] = []
    for rank in range(8, 0, -1):
        row = ""
        empty_run = 0
        for file_idx in range(8):
            name = f"{chr(ord('a') + file_idx)}{rank}"
            det = by_square.get(name)
            symbol = YOLO_TO_FEN.get(det.label) if det else None
            if not symbol:
                empty_run += 1
            else:
                if empty_run:
                    row += str(empty_run)
                    empty_run = 0
                row += symbol
        if empty_run:
            row += str(empty_run)
        ranks.append(row)
    return "/".join(ranks)


def build_yolo_square_montage(
    rectified_bgr: NDArray[np.uint8],
    assigned: dict[str, YoloPieceDetection],
    board_size: int,
) -> NDArray[np.uint8]:
    """Annotated rectified board showing mapped squares."""
    canvas = rectified_bgr.copy()
    cell = board_size // 8
    for row in range(8):
        for col in range(8):
            rank = 8 - row
            name = f"{chr(ord('a') + col)}{rank}"
            x0, y0 = col * cell, row * cell
            det = assigned.get(name)
            if det:
                color = (90, 200, 255) if det.label.startswith("white_") else (180, 120, 255)
                cv2.rectangle(canvas, (x0, y0), (x0 + cell - 1, y0 + cell - 1), color, 2)
                short = det.label.replace("_", " ")[:10]
                cv2.putText(
                    canvas,
                    f"{name} {short} {det.confidence:.2f}",
                    (x0 + 3, y0 + max(14, cell // 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return canvas
