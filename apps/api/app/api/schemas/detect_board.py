"""Detect-board API schemas."""

import base64

from pydantic import BaseModel, Field

from app.api.schemas.common import PointSchema


class SquareInfoSchema(BaseModel):
    name: str = Field(..., description="Algebraic square name, e.g. e4")
    filename: str = Field(..., description="Crop filename, e.g. e4.png")
    cell_bbox: list[int] = Field(..., min_length=4, max_length=4)
    crop_bbox: list[int] = Field(..., min_length=4, max_length=4)
    url: str
    label: str = "empty"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    occupied: bool = False


class DetectionDebugSchema(BaseModel):
    original_base64: str | None = None
    detected_lines_base64: str | None = None
    intersections_base64: str | None = None
    mesh_base64: str | None = None
    rectified_board_base64: str | None = None
    rectified_upscaled_base64: str | None = None
    square_extraction_base64: str | None = None
    crop_quality_base64: str | None = None
    occupancy_base64: str | None = None
    occupancy_detail_base64: str | None = None
    classifier_confidence_base64: str | None = None
    ml_occupancy_base64: str | None = None
    ml_piece_top1_base64: str | None = None
    ml_onnx_occ_crops_base64: str | None = None
    ml_onnx_piece_crops_base64: str | None = None
    ml_detail_base64: str | None = None
    fen_candidates_base64: str | None = None
    mesh_quality_base64: str | None = None
    final_board_base64: str | None = None
    grid_debug_extreme_base64: str | None = None
    dataset_squares_base64: str | None = None


class BoardMatrixCellSchema(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class DetectBoardResponse(BaseModel):
    job_id: str
    corners: list[PointSchema] = Field(..., min_length=4, max_length=4)
    homography: list[list[float]]
    confidence: float = Field(..., ge=0.0, le=1.0)
    original_width: int = Field(..., gt=0)
    original_height: int = Field(..., gt=0)
    output_width: int = Field(..., gt=0)
    output_height: int = Field(..., gt=0)
    warped_image_base64: str = Field(..., description="JPEG-encoded top-down board")
    squares: list[SquareInfoSchema] = Field(..., min_length=64, max_length=64)
    fen: str | None = Field(None, description="Best-effort FEN from classification (may be invalid)")
    interactive_fen: str | None = Field(None, description="Legal FEN for interactive board — null if not validated")
    fen_confidence: float | None = Field(None, ge=0.0, le=1.0)
    fen_valid: bool | None = None
    board_ready: bool = False
    board_matrix: list[list[BoardMatrixCellSchema]] | None = Field(
        None, description="Validated 8×8 piece matrix (rank 8 → rank 1)"
    )
    orientation: str | None = None
    debug: DetectionDebugSchema | None = None
    debug_overlay_base64: str | None = None
    debug_montage_base64: str | None = None
    processing_ms: int = Field(..., ge=0)
    metadata: dict

    @classmethod
    def from_detection(
        cls,
        *,
        job_id: str,
        corners: list[tuple[float, float]],
        homography: list[list[float]],
        confidence: float,
        original_width: int,
        original_height: int,
        output_width: int,
        output_height: int,
        warped_jpeg: bytes,
        squares: list[SquareInfoSchema],
        fen: str | None,
        interactive_fen: str | None,
        fen_confidence: float | None,
        fen_valid: bool | None,
        board_ready: bool,
        board_matrix: list[list[dict[str, str | float]]] | None,
        orientation: str | None,
        debug_overlay_jpeg: bytes | None,
        debug_montage_jpeg: bytes | None,
        debug_jpegs: dict[str, bytes] | None,
        processing_ms: int,
        metadata: dict,
    ) -> DetectBoardResponse:
        return cls(
            job_id=job_id,
            corners=[PointSchema(x=x, y=y) for x, y in corners],
            homography=homography,
            confidence=confidence,
            original_width=original_width,
            original_height=original_height,
            output_width=output_width,
            output_height=output_height,
            warped_image_base64=base64.b64encode(warped_jpeg).decode("ascii"),
            squares=squares,
            fen=fen,
            interactive_fen=interactive_fen,
            fen_confidence=fen_confidence,
            fen_valid=fen_valid,
            board_ready=board_ready,
            board_matrix=_map_board_matrix(board_matrix),
            orientation=orientation,
            debug=_build_debug_schema(debug_jpegs),
            debug_overlay_base64=_encode(debug_overlay_jpeg),
            debug_montage_base64=_encode(debug_montage_jpeg),
            processing_ms=processing_ms,
            metadata=metadata,
        )


def _map_board_matrix(
    raw: list[list[dict[str, str | float]]] | None,
) -> list[list[BoardMatrixCellSchema]] | None:
    if raw is None:
        return None
    return [
        [BoardMatrixCellSchema(label=str(cell["label"]), confidence=float(cell["confidence"])) for cell in row]
        for row in raw
    ]


def _build_debug_schema(debug_jpegs: dict[str, bytes] | None) -> DetectionDebugSchema | None:
    if not debug_jpegs:
        return None
    return DetectionDebugSchema(
        original_base64=_encode(debug_jpegs.get("original")),
        detected_lines_base64=_encode(debug_jpegs.get("detected_lines")),
        intersections_base64=_encode(debug_jpegs.get("intersections")),
        mesh_base64=_encode(debug_jpegs.get("mesh")),
        rectified_board_base64=_encode(debug_jpegs.get("rectified_board")),
        rectified_upscaled_base64=_encode(debug_jpegs.get("rectified_upscaled")),
        square_extraction_base64=_encode(debug_jpegs.get("square_extraction")),
        crop_quality_base64=_encode(debug_jpegs.get("crop_quality")),
        occupancy_base64=_encode(debug_jpegs.get("occupancy")),
        occupancy_detail_base64=_encode(debug_jpegs.get("occupancy_detail")),
        classifier_confidence_base64=_encode(debug_jpegs.get("classifier_confidence")),
        ml_occupancy_base64=_encode(debug_jpegs.get("ml_occupancy")),
        ml_piece_top1_base64=_encode(debug_jpegs.get("ml_piece_top1")),
        ml_onnx_occ_crops_base64=_encode(debug_jpegs.get("ml_onnx_occ_crops")),
        ml_onnx_piece_crops_base64=_encode(debug_jpegs.get("ml_onnx_piece_crops")),
        ml_detail_base64=_encode(debug_jpegs.get("ml_detail")),
        fen_candidates_base64=_encode(debug_jpegs.get("fen_candidates")),
        mesh_quality_base64=_encode(debug_jpegs.get("mesh_quality")),
        final_board_base64=_encode(debug_jpegs.get("final_board")),
        grid_debug_extreme_base64=_encode(debug_jpegs.get("grid_debug_extreme")),
        dataset_squares_base64=_encode(debug_jpegs.get("dataset_squares")),
    )


def _encode(data: bytes | None) -> str | None:
    if data is None:
        return None
    return base64.b64encode(data).decode("ascii")
