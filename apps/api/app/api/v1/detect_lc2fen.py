"""LiveChess2FEN geometry + YOLO piece detection endpoint."""

from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.api.deps import get_lc2fen_adapter, get_storage_service
from app.api.schemas.detect_board import (
    DetectBoardResponse,
    SquareInfoSchema,
)
from app.core.exceptions import BoardDetectionError
from app.core.settings import settings
from app.utils.file_storage import StorageService
from app.utils.image_validation import validate_upload
from vision.lc2fen.adapter import (
    LC2FENAdapter,
    build_square_montage,
    draw_corner_overlay,
    fen_to_board_matrix,
)
from vision.lc2fen.yolo_pieces import build_yolo_square_montage

router = APIRouter(tags=["detection"])


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode debug image")
    return buf.tobytes()


@router.post("/detect-lc2fen", response_model=DetectBoardResponse)
async def detect_lc2fen(
    file: UploadFile = File(..., description="Photo containing a chess board"),
    a1_pos: str = Query("BL", pattern="^(BL|BR|TL|TR)$", description="Location of a1 square"),
    conf_threshold: float = Query(0.30, ge=0.1, le=0.95, description="YOLO min confidence"),
    storage: StorageService = Depends(get_storage_service),
    adapter: LC2FENAdapter = Depends(get_lc2fen_adapter),
) -> DetectBoardResponse:
    """LC2FEN board geometry + YOLO piece object detection on rectified board."""
    data = await file.read()
    validated = validate_upload(
        data,
        filename=file.filename,
        content_type=file.content_type,
        settings=settings,
    )

    job_id = storage.new_scan_id()
    job_dir = storage.job_directory(job_id)
    debug_dir = job_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    squares_dir = job_dir / "squares"
    squares_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    adapter.a1_pos = a1_pos

    try:
        result = adapter.predict_yolo_from_bytes(
            validated.data,
            job_dir=job_dir,
            filename="input.jpg",
            conf_threshold=conf_threshold,
        )
    except Exception as exc:
        raise BoardDetectionError(f"LC2FEN+YOLO failed: {exc}") from exc

    geometry = result.geometry
    original_bgr = cv2.imdecode(np.frombuffer(validated.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise BoardDetectionError("Unable to decode uploaded image")

    rectified_bgr = geometry.rectified_bgr
    warped_jpeg = _encode_jpeg(geometry.warped_bgr)
    rectified_jpeg = _encode_jpeg(rectified_bgr)
    overlay_bgr = draw_corner_overlay(original_bgr, geometry.corners)
    overlay_jpeg = _encode_jpeg(overlay_bgr)
    yolo_overlay_jpeg = _encode_jpeg(result.yolo_overlay_bgr)
    board_size = min(rectified_bgr.shape[0], rectified_bgr.shape[1])
    montage_bgr = build_yolo_square_montage(rectified_bgr, result.assigned, board_size)
    montage_jpeg = _encode_jpeg(montage_bgr)

    (job_dir / "warped.jpg").write_bytes(warped_jpeg)
    (debug_dir / "original.jpg").write_bytes(_encode_jpeg(original_bgr))
    (debug_dir / "rectified_board.jpg").write_bytes(rectified_jpeg)
    (debug_dir / "grid_overlay.jpg").write_bytes(overlay_jpeg)
    (debug_dir / "yolo_overlay.jpg").write_bytes(yolo_overlay_jpeg)
    (debug_dir / "crop_montage.jpg").write_bytes(montage_jpeg)

    out_h, out_w = rectified_bgr.shape[:2]
    cell = out_h // 8
    prefix = settings.api_prefix

    squares_response: list[SquareInfoSchema] = []
    for sq in result.square_predictions:
        x0 = sq.col * cell
        y0 = sq.row * cell
        filename = f"{sq.name}.jpg"
        crop = rectified_bgr[y0 : y0 + cell, x0 : x0 + cell]
        crop_path = squares_dir / filename
        cv2.imwrite(str(crop_path), crop)
        label = sq.label if sq.occupied else "empty"
        squares_response.append(
            SquareInfoSchema(
                name=sq.name,
                filename=filename,
                cell_bbox=[x0, y0, cell, cell],
                crop_bbox=[x0, y0, cell, cell],
                url=f"{prefix}/detect-lc2fen/{job_id}/squares/{filename}",
                label=label,
                confidence=sq.confidence,
                occupied=sq.occupied,
            )
        )

    board_matrix_raw = fen_to_board_matrix(result.fen, result.square_predictions)
    processing_ms = int((time.perf_counter() - started) * 1000)
    metadata = {
        **result.metadata,
        "processing_ms": processing_ms,
        "source_filename": validated.filename,
        "source_mime_type": validated.mime_type,
        "geometry_backend": "lc2fen",
        "piece_backend": "yolo_object_detection",
        "conf_threshold": conf_threshold,
        "fen_validation": {
            "is_valid": result.validation.is_valid,
            "kings_ok": result.validation.kings_ok,
            "piece_count": result.validation.piece_count,
        },
    }

    debug = {
        "original": _encode_jpeg(original_bgr),
        "rectified_board": rectified_jpeg,
        "grid_overlay": overlay_jpeg,
        "ml_piece_top1": yolo_overlay_jpeg,
        "final_board": montage_jpeg,
        "crop_montage": montage_jpeg,
    }

    corners = [(float(c[0]), float(c[1])) for c in geometry.corners]

    return DetectBoardResponse.from_detection(
        job_id=job_id,
        corners=corners,
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        confidence=result.validation.confidence,
        original_width=geometry.original_width,
        original_height=geometry.original_height,
        output_width=out_w,
        output_height=out_h,
        warped_jpeg=warped_jpeg,
        squares=squares_response,
        fen=result.fen,
        interactive_fen=result.validation.interactive_fen,
        fen_confidence=result.validation.confidence,
        fen_valid=result.validation.is_valid,
        board_ready=result.validation.board_ready,
        board_matrix=board_matrix_raw,
        orientation="standard",
        debug_overlay_jpeg=overlay_jpeg,
        debug_montage_jpeg=yolo_overlay_jpeg,
        debug_jpegs=debug,
        processing_ms=processing_ms,
        metadata=metadata,
    )


@router.get("/detect-lc2fen/{job_id}/squares/{filename}")
async def get_lc2fen_square_crop(
    job_id: str,
    filename: str,
    storage: StorageService = Depends(get_storage_service),
):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    path = storage.job_directory(job_id) / "squares" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Square crop not found")
    return FileResponse(path, media_type="image/jpeg")
