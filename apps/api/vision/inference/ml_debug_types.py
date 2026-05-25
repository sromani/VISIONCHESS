"""Structured ML debug payloads — logits, top-k, ONNX input crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ClassPrediction:
    label: str
    probability: float
    logit: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "probability": self.probability, "logit": self.logit}


@dataclass(frozen=True, slots=True)
class OccupancyMlDebug:
    occupied_probability: float
    empty_probability: float
    logits: tuple[float, ...]
    model_input_size: int
    onnx_input_bgr: NDArray[np.uint8]

    def to_dict(self) -> dict[str, Any]:
        return {
            "occupied_probability": self.occupied_probability,
            "empty_probability": self.empty_probability,
            "logits": list(self.logits),
            "model_input_size": self.model_input_size,
        }


@dataclass(frozen=True, slots=True)
class PieceMlDebug:
    top3: tuple[ClassPrediction, ...]
    logits: tuple[float, ...]
    class_names: tuple[str, ...]
    model_input_size: int
    onnx_input_bgr: NDArray[np.uint8]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top3": [p.to_dict() for p in self.top3],
            "logits": list(self.logits),
            "class_names": list(self.class_names),
            "model_input_size": self.model_input_size,
        }


@dataclass(frozen=True, slots=True)
class SquareMlDebug:
    square_name: str
    row: int
    col: int
    analysis_crop_shape: tuple[int, int]
    occupancy: OccupancyMlDebug | None
    piece: PieceMlDebug | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "square_name": self.square_name,
            "row": self.row,
            "col": self.col,
            "analysis_crop_shape": list(self.analysis_crop_shape),
            "occupancy": None if self.occupancy is None else self.occupancy.to_dict(),
            "piece": None if self.piece is None else self.piece.to_dict(),
        }


@dataclass
class MlDebugReport:
    piece_model: str
    occupancy_model: str
    squares: dict[str, SquareMlDebug]

    def to_dict(self) -> dict[str, Any]:
        return {
            "piece_model": self.piece_model,
            "occupancy_model": self.occupancy_model,
            "squares": {name: sq.to_dict() for name, sq in sorted(self.squares.items())},
        }
