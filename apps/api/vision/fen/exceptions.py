"""FEN builder errors."""

from __future__ import annotations


class FenBuildError(Exception):
    """Base error for FEN construction."""


class InvalidGridError(FenBuildError):
    """Input grid is not a well-formed 8×8 detection matrix."""
