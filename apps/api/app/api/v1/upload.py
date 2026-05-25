"""Image upload endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_scan_service
from app.api.mappers.scan import scan_to_response
from app.api.schemas.scan import UploadResponse
from app.core.settings import settings
from app.services.scan_service import ScanService
from app.utils.image_validation import validate_upload

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(..., description="Chess board photo (JPEG, PNG, WebP)"),
    scan_service: ScanService = Depends(get_scan_service),
) -> UploadResponse:
    """Upload a chess board image and run the vision pipeline."""
    data = await file.read()
    validated = validate_upload(
        data,
        filename=file.filename,
        content_type=file.content_type,
        settings=settings,
    )
    record = await scan_service.process_upload(validated)
    return scan_to_response(record, settings.api_prefix)
