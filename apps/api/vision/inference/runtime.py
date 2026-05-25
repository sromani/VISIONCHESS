"""Unified ONNX inference runtime for occupancy + piece models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vision.inference.model_registry import resolve_occupancy_model, resolve_piece_model

if TYPE_CHECKING:
    from vision.occupancy.ml_model import MlOccupancyModel


@dataclass(frozen=True, slots=True)
class InferenceRuntime:
    """Loads production ONNX models — fails fast if required models missing."""

    occupancy_model: MlOccupancyModel | None
    piece_model_path: Path | None
    occupancy_available: bool
    piece_available: bool

    @classmethod
    def load(cls, *, piece_path: str | None = None, occupancy_path: Path | None = None) -> InferenceRuntime:
        piece_art = resolve_piece_model(piece_path)
        occ_art = resolve_occupancy_model(occupancy_path)

        occ_model = None
        occ_ok = occ_art is not None
        if occ_art is not None:
            try:
                from vision.occupancy.ml_model import MlOccupancyConfig, MlOccupancyModel

                occ_model = MlOccupancyModel(
                    MlOccupancyConfig(model_path=occ_art.path, image_size=occ_art.image_size)
                )
            except Exception:
                occ_model = None
                occ_ok = False

        piece_ok = piece_art is not None
        return cls(
            occupancy_model=occ_model,
            piece_model_path=piece_art.path if piece_art else None,
            occupancy_available=occ_ok,
            piece_available=piece_ok,
        )

    def require_piece(self) -> None:
        if not self.piece_available:
            msg = (
                "Required piece classifier ONNX missing. "
                "Run: cd ml && python scripts/export_chesscog_standalone.py"
            )
            raise RuntimeError(msg)

    def require_all(self) -> None:
        missing: list[str] = []
        if not self.occupancy_available:
            missing.append("occupancy.onnx")
        if not self.piece_available:
            missing.append("piece_classifier.onnx")
        if missing:
            msg = (
                f"Required ML models missing: {', '.join(missing)}. "
                "Run: cd ml && python scripts/setup_pretrained.py --source synthetic"
            )
            raise RuntimeError(msg)
