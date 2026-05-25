"""Shared API schemas."""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str
    message: str


class PointSchema(BaseModel):
    x: float
    y: float


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    environment: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
