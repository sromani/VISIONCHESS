"""Scan domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ScanStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScanRecord:
    id: str
    status: ScanStatus
    original_path: Path
    original_filename: str
    mime_type: str
    warped_path: Path | None = None
    fen: str | None = None
    fen_confidence: float | None = None
    board_corners: list[dict[str, float]] | None = None
    grid_metadata: dict[str, Any] | None = None
    error_message: str | None = None
    processing_ms: int | None = None
    pipeline_timing: dict[str, int] | None = field(default=None)
