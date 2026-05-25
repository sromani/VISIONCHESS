"""Piece classifier backends — ML-only in production."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision.classification.backend_protocol import PieceClassifierBackend
from vision.classification.types import SquareClassification
from vision.inference.model_registry import resolve_piece_model
from vision.inference.piece_pipeline import PieceInferenceConfig, PieceInferencePipeline


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    backend: str = "auto"
    model_path: Path | None = None
    min_confidence: float = 0.38
    use_tta: bool = False
    allow_heuristic_fallback: bool = False


def create_classifier(config: ClassifierConfig | None = None) -> PieceClassifierBackend:
    cfg = config or ClassifierConfig()
    artifact = resolve_piece_model(cfg.model_path)

    if artifact is not None and cfg.backend in {"auto", "onnx", "ml"}:
        return PieceInferencePipeline(
            artifact,
            PieceInferenceConfig(model_path=artifact.path, use_tta=cfg.use_tta),
        )

    if cfg.backend == "heuristic" and cfg.allow_heuristic_fallback:
        from vision.classification.heuristic import HeuristicClassifierConfig
        from vision.classification.classifier_legacy import HeuristicBackend

        return HeuristicBackend(HeuristicClassifierConfig())

    msg = (
        "Piece classifier ONNX not found. Bootstrap with:\n"
        "  cd ml && python scripts/setup_pretrained.py --source synthetic\n"
        "Or import chesscog weights:\n"
        "  cd ml && python scripts/export_chesscog_standalone.py\n"
        "  python scripts/setup_pretrained.py --source lc2fen --lc2fen-dir PATH"
    )
    raise RuntimeError(msg)


def resolve_model_path(explicit: str | None = None) -> Path | None:
    artifact = resolve_piece_model(explicit)
    return artifact.path if artifact else None
