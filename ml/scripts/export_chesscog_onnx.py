"""Export chesscog pretrained PyTorch models to VisionChess ONNX format."""

from __future__ import annotations

import json
from pathlib import Path

import torch

DEFAULT_CLASSES = (
    "empty",
    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
)


def export_all(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_piece_classifier(output_dir / "chesscog_piece.onnx")
    export_occupancy_classifier(output_dir / "chesscog_occupancy.onnx")
    piece_dest = output_dir.parent / "piece_classifier" / "piece_classifier.onnx"
    piece_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil_copy(output_dir / "chesscog_piece.onnx", piece_dest)
    shutil_copy(output_dir / "chesscog_piece.json", piece_dest.with_suffix(".json"))


def export_piece_classifier(out_path: Path) -> None:
    model, image_size = _load_chesscog_piece_model()
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["square"],
        output_names=["logits"],
        dynamic_axes={"square": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    meta = {
        "source": "chesscog",
        "image_size": image_size,
        "num_classes": 13,
        "class_names": list(DEFAULT_CLASSES),
        "input_name": "square",
        "output_name": "logits",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported chesscog piece -> {out_path}")


def export_occupancy_classifier(out_path: Path) -> None:
    model, image_size = _load_chesscog_occupancy_model()
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["square"],
        output_names=["logits"],
        dynamic_axes={"square": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    meta = {
        "source": "chesscog",
        "image_size": image_size,
        "num_classes": 2,
        "class_names": ["empty", "occupied"],
        "input_name": "square",
        "output_name": "logits",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported chesscog occupancy -> {out_path}")


def _load_chesscog_piece_model():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "chesscog.piece_classifier.download_model"])
    from chesscog.core.models import build_model
    from chesscog.core.registry import Registry

    registry = Registry("models://piece_classifier")
    model = build_model(registry, training=False)
    image_size = getattr(model, "input_size", (100, 100))
    if isinstance(image_size, tuple):
        image_size = image_size[0]
    return model, int(image_size)


def _load_chesscog_occupancy_model():
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "chesscog.occupancy_classifier.download_model"])
    from chesscog.core.models import build_model
    from chesscog.core.registry import Registry

    registry = Registry("models://occupancy_classifier")
    model = build_model(registry, training=False)
    image_size = getattr(model, "input_size", (100, 100))
    if isinstance(image_size, tuple):
        image_size = image_size[0]
    return model, int(image_size)


def shutil_copy(src: Path, dest: Path) -> None:
    import shutil

    shutil.copy2(src, dest)
    meta = src.with_suffix(".json")
    if meta.exists():
        shutil.copy2(meta, dest.with_suffix(".json"))


if __name__ == "__main__":
    export_all(Path(__file__).resolve().parents[1] / "models" / "pretrained")
