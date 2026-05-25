"""Analysis routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_analysis_service
from app.api.schemas.analysis import AnalysisRequest, AnalysisResponse, EngineLineSchema
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisResponse)
async def analyze_position(
    request: AnalysisRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    result = await analysis_service.analyze(request.fen, request.depth, request.multipv)
    return AnalysisResponse(
        id=result.id,
        fen=result.fen,
        depth=result.depth,
        evaluation_cp=result.evaluation_cp,
        evaluation_mate=result.evaluation_mate,
        best_move=result.best_move,
        lines=[
            EngineLineSchema(
                move=line.move,
                eval_cp=line.eval_cp,
                eval_mate=line.eval_mate,
                pv=line.pv,
            )
            for line in result.lines
        ],
        processing_ms=result.processing_ms,
    )
