"""Scanner operating modes."""

from __future__ import annotations

from enum import StrEnum


class ScannerMode(StrEnum):
    FULL = "full"
    PIECE_DETECTION_ONLY = "piece_detection_only"
