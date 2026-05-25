"""Temporary file storage for uploads."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.core.logging import get_logger
from app.core.settings import Settings

logger = get_logger(__name__)


class StorageService:
    """Persists uploaded images and pipeline artifacts to local disk."""

    def __init__(self, settings: Settings) -> None:
        self._root = settings.storage_path
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def save_original(self, scan_id: str, data: bytes, extension: str) -> Path:
        path = self._root / f"{scan_id}_original{extension}"
        path.write_bytes(data)
        logger.info("stored_original", scan_id=scan_id, path=str(path), bytes=len(data))
        return path

    def save_warped(self, scan_id: str, data: bytes) -> Path:
        path = self._root / f"{scan_id}_warped.jpg"
        path.write_bytes(data)
        logger.info("stored_warped", scan_id=scan_id, path=str(path), bytes=len(data))
        return path

    def extension_for_mime(self, mime_type: str) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(mime_type, ".bin")

    def new_scan_id(self) -> str:
        return str(uuid.uuid4())

    def job_directory(self, job_id: str) -> Path:
        path = self._root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def squares_directory(self, job_id: str) -> Path:
        path = self.job_directory(job_id) / "squares"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def debug_directory(self, job_id: str) -> Path:
        path = self.job_directory(job_id) / "debug"
        path.mkdir(parents=True, exist_ok=True)
        return path
