"""Analysis API schemas."""

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    fen: str = Field(..., min_length=10)
    depth: int = Field(default=18, ge=1, le=30)
    multipv: int = Field(default=3, ge=1, le=5)


class EngineLineSchema(BaseModel):
    move: str
    eval_cp: int | None = None
    eval_mate: int | None = None
    pv: list[str]


class AnalysisResponse(BaseModel):
    id: str
    fen: str
    depth: int
    evaluation_cp: int | None
    evaluation_mate: int | None
    best_move: str
    lines: list[EngineLineSchema]
    processing_ms: int
