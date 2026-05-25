"""Image validation tests."""

import pytest

from app.core.exceptions import ImageValidationError
from app.core.settings import Settings
from app.utils.image_validation import detect_mime_type, validate_upload


@pytest.fixture
def settings() -> Settings:
    return Settings(max_upload_bytes=1024 * 1024)


def test_detect_jpeg() -> None:
    assert detect_mime_type(b"\xff\xd8\xff\xe0\x00") == "image/jpeg"


def test_detect_png() -> None:
    assert detect_mime_type(b"\x89PNG\r\n\x1a\n") == "image/png"


def test_rejects_empty(settings: Settings) -> None:
    with pytest.raises(ImageValidationError):
        validate_upload(b"", filename="x.jpg", content_type="image/jpeg", settings=settings)


def test_rejects_oversized(settings: Settings) -> None:
    data = b"\xff\xd8\xff" + b"\x00" * 2_000_000
    with pytest.raises(ImageValidationError):
        validate_upload(data, filename="big.jpg", content_type="image/jpeg", settings=settings)


def test_accepts_valid_jpeg(settings: Settings) -> None:
    data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    result = validate_upload(data, filename="board.jpg", content_type="image/jpeg", settings=settings)
    assert result.mime_type == "image/jpeg"
    assert result.size_bytes == len(data)
