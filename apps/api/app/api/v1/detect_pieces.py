"""Piece-detection-only endpoint — no FEN, no occupancy heuristics."""

from __future__ import annotations

import time

import cv2
from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_piece_detection_pipeline, get_storage_service
from app.api.schemas.piece_detection import PieceDetectionResponse
from app.core.exceptions import BoardDetectionError
from app.core.settings import settings
from app.utils.file_storage import StorageService
from app.utils.image_validation import validate_upload
from vision.board.exceptions import BoardNotFoundError
from vision.scanner.piece_detection_pipeline import PieceDetectionPipeline

router = APIRouter(tags=["detection"])


@router.post("/detect-pieces", response_model=PieceDetectionResponse)
async def detect_pieces(
    file: UploadFile = File(..., description="Photo containing a chess board"),
    pipeline: PieceDetectionPipeline = Depends(get_piece_detection_pipeline),
    storage: StorageService = Depends(get_storage_service),
) -> PieceDetectionResponse:
    """Geometry + ML piece ONNX only. Fails if no piece classifier ONNX."""
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

    started = time.perf_counter()
    try:
        result = pipeline.run_from_bytes(validated.data)
    except BoardNotFoundError as exc:
        raise BoardDetectionError(str(exc)) from exc
    except RuntimeError as exc:
        raise BoardDetectionError(str(exc)) from exc

    ok, buf = cv2.imencode(".jpg", result.rectified_board, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        (job_dir / "rectified.jpg").write_bytes(buf.tobytes())

    debug_jpegs = dict(result.debug_jpegs)
    for name, jpeg in debug_jpegs.items():
        (debug_dir / f"{name}.jpg").write_bytes(jpeg)

    processing_ms = int((time.perf_counter() - started) * 1000)
    metadata = dict(result.metadata)
    metadata["processing_ms"] = processing_ms
    metadata["source_filename"] = validated.filename

    return PieceDetectionResponse.from_result(
        job_id=job_id,
        result=result,
        debug_jpegs=debug_jpegs,
        processing_ms=processing_ms,
    )
