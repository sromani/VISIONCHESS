"""Scan retrieval and asset routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_scan_service
from app.api.mappers.scan import scan_to_response
from app.api.schemas.scan import ScanResponse, UploadResponse
from app.core.exceptions import NotFoundError
from app.core.settings import settings
from app.services.scan_service import ScanService
from app.utils.image_validation import validate_upload

router = APIRouter(prefix="/scans", tags=["scans"])


@router.post("", response_model=UploadResponse, status_code=201)
async def create_scan(
    file: UploadFile = File(...),
    scan_service: ScanService = Depends(get_scan_service),
) -> UploadResponse:
    """Create a scan from an uploaded image."""
    data = await file.read()
    validated = validate_upload(
        data,
        filename=file.filename,
        content_type=file.content_type,
        settings=settings,
    )
    record = await scan_service.process_upload(validated)
    return scan_to_response(record, settings.api_prefix)


@router.get("/{scan_id}", response_model=ScanResponse)
async def get_scan(
    scan_id: str,
    scan_service: ScanService = Depends(get_scan_service),
) -> ScanResponse:
    record = scan_service.get(scan_id)
    return scan_to_response(record, settings.api_prefix)


@router.get("/{scan_id}/image")
async def get_scan_image(
    scan_id: str,
    variant: str = "original",
    scan_service: ScanService = Depends(get_scan_service),
) -> FileResponse:
    record = scan_service.get(scan_id)
    path = record.warped_path if variant == "warped" else record.original_path
    if path is None or not path.exists():
        raise NotFoundError("Image not found")

    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media = media_types.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)
