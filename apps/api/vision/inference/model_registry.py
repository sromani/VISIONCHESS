"""Resolve and load production piece/occupancy ONNX models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ModelSource = Literal["visionchess", "chesscog", "lc2fen", "chess_vision", "nakst_yolov8m", "yamero_yolo11n", "custom"]

YOLO_MODEL_ALIASES: dict[str, str] = {
    "nakst": "pretrained/yolov8_chess_pieces.onnx",
    "yolo11n": "pretrained/yolo11n_chess_pieces.onnx",
}


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    path: Path
    source: ModelSource
    image_size: int
    num_classes: int
    class_names: tuple[str, ...]
    input_name: str
    output_name: str
    includes_empty_class: bool = True
    class_map: dict[str, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]

    @classmethod
    def from_onnx(cls, path: Path, *, source: ModelSource = "custom") -> ModelArtifact:
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_map = data.get("class_map")
            class_map = {str(k): str(v) for k, v in raw_map.items()} if isinstance(raw_map, dict) else None
            known = {
                "source", "image_size", "num_classes", "class_names", "input_name",
                "output_name", "includes_empty_class", "class_map", "model_type",
            }
            extra = {k: v for k, v in data.items() if k not in known}
            return cls(
                path=path,
                source=data.get("source", source),  # type: ignore[arg-type]
                image_size=int(data.get("image_size", 64)),
                num_classes=int(data.get("num_classes", 13)),
                class_names=tuple(data.get("class_names", _default_class_names())),
                input_name=str(data.get("input_name", "square")),
                output_name=str(data.get("output_name", "logits")),
                includes_empty_class=bool(data.get("includes_empty_class", True)),
                class_map=class_map,
                extra=extra,
            )
        return cls(
            path=path,
            source=source,
            image_size=64,
            num_classes=13,
            class_names=_default_class_names(),
            input_name="square",
            output_name="logits",
            includes_empty_class=True,
            class_map=None,
            extra={},
        )

    def label_for_index(self, index: int) -> str:
        if index < 0 or index >= len(self.class_names):
            return "empty"
        raw = self.class_names[index]
        if self.class_map is not None:
            return self.class_map.get(raw, raw)
        return raw


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def piece_model_candidates(explicit: str | Path | None = None) -> list[Path]:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return [p]
    root = repo_root() / "ml" / "models"
    names = [
        "pretrained/chesscog_piece.onnx",
        "piece_classifier/best.onnx",
        "piece_classifier/piece_classifier.onnx",
        "pretrained/lc2fen_piece.onnx",
        "pretrained/chess_vision_piece.onnx",
        "pretrained/piece.onnx",
    ]
    return [root / n for n in names if (root / n).exists()]


def occupancy_model_candidates(explicit: Path | None = None) -> list[Path]:
    if explicit and explicit.exists():
        return [explicit]
    root = repo_root() / "ml" / "models"
    names = [
        "pretrained/chesscog_occupancy.onnx",
        "occupancy/occupancy.onnx",
        "occupancy/best.onnx",
        "pretrained/occupancy.onnx",
    ]
    return [root / n for n in names if (root / n).exists()]


def yolo_model_candidates(explicit: str | Path | None = None) -> list[Path]:
    if explicit:
        key = str(explicit).strip()
        root = repo_root() / "ml" / "models"
        if key in YOLO_MODEL_ALIASES:
            path = root / YOLO_MODEL_ALIASES[key]
            if path.exists():
                return [path]
        p = Path(key)
        if p.exists():
            return [p]
        alias_path = root / key
        if alias_path.exists():
            return [alias_path]
    root = repo_root() / "ml" / "models"
    import os

    preferred = os.environ.get("VISIONCHESS_YOLO_MODEL", "nakst").strip()
    if preferred in YOLO_MODEL_ALIASES:
        path = root / YOLO_MODEL_ALIASES[preferred]
        if path.exists():
            return [path]
    names = [
        "pretrained/yolov8_chess_pieces.onnx",
        "pretrained/yolo11n_chess_pieces.onnx",
        "pretrained/yolo_chess_pieces.onnx",
    ]
    found = [root / n for n in names if (root / n).exists()]
    if found:
        return found
    hf_glob = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_glob.exists():
        for path in hf_glob.glob("models--NAKSTStudio--yolov8m-chess-piece-detection/snapshots/*/best.onnx"):
            return [path]
    return []


def resolve_yolo_model(explicit: str | Path | None = None) -> ModelArtifact | None:
    for path in yolo_model_candidates(explicit):
        if "yolo11n" in path.name:
            source: ModelSource = "yamero_yolo11n"
        elif "yolov8" in path.name or "NAKST" in str(path):
            source = "nakst_yolov8m"
        else:
            source = "custom"
        return ModelArtifact.from_onnx(path, source=source)
    return None


def resolve_piece_model(explicit: str | Path | None = None) -> ModelArtifact | None:
    for path in piece_model_candidates(explicit):
        return ModelArtifact.from_onnx(path)
    return None


def resolve_occupancy_model(explicit: Path | None = None) -> ModelArtifact | None:
    for path in occupancy_model_candidates(explicit):
        source: ModelSource = "visionchess"
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            source = data.get("source", source)  # type: ignore[assignment]
        return ModelArtifact.from_onnx(path, source=source)
    return None


def _default_class_names() -> tuple[str, ...]:
    return (
        "empty",
        "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
        "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
    )
