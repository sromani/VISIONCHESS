"""Image upload validation."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import ImageValidationError
from app.core.settings import Settings

MAGIC_SIGNATURES: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # verified further below
}


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    data: bytes
    mime_type: str
    filename: str
    size_bytes: int


def validate_upload(
    data: bytes,
    *,
    filename: str | None,
    content_type: str | None,
    settings: Settings,
) -> ValidatedImage:
    if not data:
        raise ImageValidationError("Empty file")

    if len(data) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // (1024 * 1024)
        raise ImageValidationError(f"File exceeds {max_mb} MB limit")

    detected = detect_mime_type(data)
    if detected is None:
        raise ImageValidationError("Unsupported or invalid image format")

    if detected not in settings.allowed_mime_types:
        raise ImageValidationError(f"MIME type '{detected}' is not allowed")

    if content_type and content_type not in settings.allowed_mime_types:
        raise ImageValidationError(f"Declared content type '{content_type}' is not allowed")

    safe_name = (filename or "upload.jpg").replace("\\", "_").replace("/", "_")
    return ValidatedImage(
        data=data,
        mime_type=detected,
        filename=safe_name,
        size_bytes=len(data),
    )


def detect_mime_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
