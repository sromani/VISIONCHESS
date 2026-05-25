"""Map domain models to API responses."""

from app.api.schemas.common import PointSchema
from app.api.schemas.scan import ScanResponse
from app.models.scan import ScanRecord


def scan_to_response(record: ScanRecord, api_prefix: str = "/api/v1") -> ScanResponse:
    warped_url = f"{api_prefix}/scans/{record.id}/image?variant=warped" if record.warped_path else None
    original_url = f"{api_prefix}/scans/{record.id}/image?variant=original"

    corners = (
        [PointSchema(x=c["x"], y=c["y"]) for c in record.board_corners]
        if record.board_corners
        else None
    )

    timing = None
    if record.pipeline_timing:
        timing = {k: v for k, v in record.pipeline_timing.items() if k.endswith("_ms")}

    return ScanResponse(
        id=record.id,
        status=record.status.value,
        fen=record.fen,
        fen_confidence=record.fen_confidence,
        warped_image_url=warped_url,
        original_image_url=original_url,
        board_corners=corners,
        grid=record.grid_metadata,
        error_message=record.error_message,
        processing_ms=record.processing_ms,
        pipeline_timing_ms=timing,
    )
