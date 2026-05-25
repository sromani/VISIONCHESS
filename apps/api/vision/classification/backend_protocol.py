"""Classifier backend protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vision.classification.types import SquareClassification


class PieceClassifierBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def classify_squares(
        self,
        squares: list,
        *,
        soft: bool = True,
    ) -> list[SquareClassification]: ...
