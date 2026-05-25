"""Backend registry for swappable ML/localization implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BackendKind = Literal["local", "chesscog", "lc2fen"]


@dataclass
class BackendRegistry:
    """Select which open-source integration backs each ML stage."""

    occupancy: BackendKind = "local"
    piece_classifier: BackendKind = "local"
    localization_fallback: BackendKind = "local"
    chesscog_models_dir: str | None = None
    lc2fen_onnx_path: str | None = None
    notes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def production(cls) -> BackendRegistry:
        return cls(
            occupancy="local",
            piece_classifier="local",
            notes={
                "chesscog": "Optional: pip install chesscog + download models; wire via chesscog_adapter",
                "lc2fen": "Optional: place LiveChess2FEN ONNX in ml/models/lc2fen/",
            },
        )
