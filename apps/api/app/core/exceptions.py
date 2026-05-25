"""Application-level exceptions and HTTP mapping."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.schemas.common import ErrorResponse


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, *, code: str = "app_error") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class ImageValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="image_validation_error")


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, code="not_found")


class BoardDetectionError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="board_not_found")


class EngineUnavailableError(AppError):
    def __init__(self, message: str = "Chess engine unavailable") -> None:
        super().__init__(message, code="engine_unavailable")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        status_code = _status_for(exc)
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(error=exc.code, message=exc.message).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        import structlog

        structlog.get_logger("app").exception("unhandled_error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred",
            ).model_dump(),
        )


def _status_for(exc: AppError) -> int:
    mapping: dict[str, int] = {
        "image_validation_error": status.HTTP_400_BAD_REQUEST,
        "not_found": status.HTTP_404_NOT_FOUND,
        "board_not_found": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "engine_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    return mapping.get(exc.code, status.HTTP_400_BAD_REQUEST)
