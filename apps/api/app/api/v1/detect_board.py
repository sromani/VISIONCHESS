"""Board detection and square-split endpoints."""

from __future__ import annotations

import re
import time

import cv2
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_scan_pipeline, get_storage_service
from app.api.schemas.detect_board import DetectBoardResponse, SquareInfoSchema
from app.core.exceptions import BoardDetectionError
from app.core.settings import settings
from app.utils.file_storage import StorageService
from app.utils.image_validation import validate_upload
from vision.board.exceptions import BoardNotFoundError
from vision.classification.square_quality import render_dataset_montage
from vision.scanner import ScanPipeline

router = APIRouter(tags=["detection"])

_SQUARE_NAME = re.compile(r"^[a-h][1-8]$")

_DEBUG_VARIANTS = {
    "original",
    "detected_lines",
    "intersections",
    "mesh",
    "rectified_board",
    "rectified_upscaled",
    "square_extraction",
    "crop_quality",
    "occupancy",
    "occupancy_detail",
    "classifier_confidence",
    "ml_occupancy",
    "ml_piece_top1",
    "ml_onnx_occ_crops",
    "ml_onnx_piece_crops",
    "ml_detail",
    "fen_candidates",
    "mesh_quality",
    "final_board",
    "grid_overlay",
    "crop_montage",
    "grid_debug_extreme",
    "dataset_squares",
    "warped",
}


def _encode_jpeg(image) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode debug image")
    return buf.tobytes()


@router.post("/detect-board", response_model=DetectBoardResponse)
async def detect_board(
    file: UploadFile = File(..., description="Photo containing a chess board"),
    dataset_mode: bool = Query(False, description="Save square crops to dataset/<job_id>/"),
    pipeline: ScanPipeline = Depends(get_scan_pipeline),
    storage: StorageService = Depends(get_storage_service),
) -> DetectBoardResponse:
    """Grid-driven scan: localize → mesh → extract → classify → validate FEN."""
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
        scan = pipeline.run_from_bytes(
            validated.data,
            job_id=job_id,
            persist_dir=job_dir,
            dataset_mode=dataset_mode,
        )
    except BoardNotFoundError as exc:
        raise BoardDetectionError(str(exc)) from exc

    classification = scan.classification
    split_result = scan.split
    if split_result is None:
        raise BoardDetectionError("Square split failed")

    warped_jpeg = _encode_jpeg(scan.warped_board)
    (job_dir / "warped.jpg").write_bytes(warped_jpeg)

    debug_jpegs = dict(scan.debug_jpegs)
    for name, jpeg in debug_jpegs.items():
        (debug_dir / f"{name}.jpg").write_bytes(jpeg)

    if classification.dataset_grid is not None:
        dataset_montage = render_dataset_montage(classification.dataset_grid)
        dataset_jpeg = _encode_jpeg(dataset_montage)
        debug_jpegs["dataset_squares"] = dataset_jpeg
        (debug_dir / "dataset_squares.jpg").write_bytes(dataset_jpeg)

    overlay_jpeg = _encode_jpeg(split_result.debug_overlay)
    montage_jpeg = _encode_jpeg(split_result.debug_montage)
    extreme_jpeg = _encode_jpeg(split_result.debug_extreme)
    (debug_dir / "grid_overlay.jpg").write_bytes(overlay_jpeg)
    (debug_dir / "crop_montage.jpg").write_bytes(montage_jpeg)
    (debug_dir / "grid_debug_extreme.jpg").write_bytes(extreme_jpeg)
    debug_jpegs["grid_debug_extreme"] = extreme_jpeg

    processing_ms = int((time.perf_counter() - started) * 1000)
    cls_by_name = {sq.square_name: sq for sq in classification.squares}
    prefix = settings.api_prefix
    squares_response = [
        SquareInfoSchema(
            name=sq.name,
            filename=sq.filename,
            cell_bbox=list(sq.cell_bbox),
            crop_bbox=list(sq.crop_bbox),
            url=f"{prefix}/detect-board/{job_id}/squares/{sq.filename}",
            label=cls_by_name[sq.name].label if sq.name in cls_by_name else "empty",
            confidence=cls_by_name[sq.name].confidence if sq.name in cls_by_name else 0.0,
            occupied=cls_by_name[sq.name].occupied if sq.name in cls_by_name else False,
        )
        for sq in split_result.saved_squares
    ]

    metadata = dict(scan.metadata)
    metadata["processing_ms"] = processing_ms
    metadata["source_filename"] = validated.filename
    metadata["source_mime_type"] = validated.mime_type
    metadata["split"] = split_result.to_metadata()
    metadata["classification"] = classification.to_metadata()
    metadata["rectification_method"] = metadata.get("rectification", {}).get("method", "piecewise_mesh")

    return DetectBoardResponse.from_detection(
        job_id=job_id,
        corners=scan.corners_list,
        homography=scan.homography_list(),
        confidence=scan.confidence,
        original_width=scan.original_width,
        original_height=scan.original_height,
        output_width=scan.output_width,
        output_height=scan.output_height,
        warped_jpeg=warped_jpeg,
        squares=squares_response,
        fen=classification.fen,
        interactive_fen=classification.interactive_fen,
        fen_confidence=classification.confidence,
        fen_valid=classification.is_valid,
        board_ready=classification.board_ready,
        board_matrix=classification.board_matrix,
        orientation=classification.orientation,
        debug_overlay_jpeg=overlay_jpeg,
        debug_montage_jpeg=montage_jpeg,
        debug_jpegs=debug_jpegs,
        processing_ms=processing_ms,
        metadata=metadata,
    )


@router.get("/detect-board/{job_id}/squares/{filename}")
async def get_square_crop(
    job_id: str,
    filename: str,
    storage: StorageService = Depends(get_storage_service),
) -> FileResponse:
    from fastapi import HTTPException

    name = filename.removesuffix(".png")
    if not _SQUARE_NAME.match(name):
        raise HTTPException(status_code=400, detail="Invalid square filename")

    path = storage.job_directory(job_id) / "squares" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Square crop not found")
    return FileResponse(path, media_type="image/png")


@router.get("/detect-board/{job_id}/debug/{variant}")
async def get_debug_image(
    job_id: str,
    variant: str,
    storage: StorageService = Depends(get_storage_service),
) -> FileResponse:
    from fastapi import HTTPException

    if variant not in _DEBUG_VARIANTS:
        raise HTTPException(status_code=400, detail=f"variant must be one of {sorted(_DEBUG_VARIANTS)}")

    if variant == "warped":
        path = storage.job_directory(job_id) / "warped.jpg"
    else:
        path = storage.job_directory(job_id) / "debug" / f"{variant}.jpg"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Debug image not found")
    return FileResponse(path, media_type="image/jpeg")
