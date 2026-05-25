"""Health check routes."""

from fastapi import APIRouter

from app.api.schemas.common import HealthResponse, ReadyResponse
from app.core.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness() -> ReadyResponse:
    storage_ok = settings.storage_path.exists()
    return ReadyResponse(
        ready=storage_ok,
        checks={"storage": storage_ok},
    )
