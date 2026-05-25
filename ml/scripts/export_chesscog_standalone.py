#!/usr/bin/env python3
"""Download chesscog weights and export to VisionChess ONNX (no chesscog pip install)."""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

ML_ROOT = Path(__file__).resolve().parents[1]
LOADER = ML_ROOT / "chesscog_loader"
sys.path.insert(0, str(LOADER))

RAW = ML_ROOT / "models" / "pretrained" / "chesscog_raw"
OUT = ML_ROOT / "models" / "pretrained"
PIECE_OUT = ML_ROOT / "models" / "piece_classifier"
OCC_OUT = ML_ROOT / "models" / "occupancy"

PIECE_URL = "https://github.com/georg-wolflein/chesscog/releases/download/0.1.0/piece_classifier.zip"
OCC_URL = "https://github.com/georg-wolflein/chesscog/releases/download/0.1.0/occupancy_classifier.zip"

CHESSCOG_PIECE_CLASSES = (
    "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
    "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook",
)

VISIONCHESS_PIECE_CLASSES = (
    "empty",
    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    PIECE_OUT.mkdir(parents=True, exist_ok=True)
    OCC_OUT.mkdir(parents=True, exist_ok=True)

    piece_pt = _ensure_download("piece_classifier.zip", PIECE_URL)
    occ_pt = _ensure_download("occupancy_classifier.zip", OCC_URL)

    print("Loading chesscog InceptionV3 piece weights...")
    piece_model = _load_piece_model(piece_pt)
    piece_onnx = OUT / "chesscog_piece.onnx"
    _export_onnx(piece_model, piece_onnx, image_size=299, num_classes=12, source="chesscog")
    _write_piece_metadata(piece_onnx)

    print("Loading chesscog ResNet occupancy weights...")
    occ_model = _load_occupancy_model(occ_pt)
    occ_onnx = OUT / "chesscog_occupancy.onnx"
    _export_onnx(occ_model, occ_onnx, image_size=100, num_classes=2, source="chesscog")
    _write_occ_metadata(occ_onnx)

    import shutil

    for src in (piece_onnx, occ_onnx):
        dest_dir = PIECE_OUT if "piece" in src.name else OCC_OUT
        name = "piece_classifier.onnx" if "piece" in src.name else "occupancy.onnx"
        shutil.copy2(src, dest_dir / name)
        shutil.copy2(src.with_suffix(".json"), dest_dir / name.replace(".onnx", ".json"))
        print(f"Installed production copy -> {dest_dir / name}")


def _ensure_download(zip_name: str, url: str) -> Path:
    zip_path = RAW / zip_name
    folder = RAW / zip_name.replace(".zip", "")
    if zip_name.startswith("piece"):
        pt = folder / "InceptionV3.pt"
    else:
        pt = folder / "ResNet.pt"
    if pt.exists():
        return pt

    if not zip_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)

    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(folder)
    if not pt.exists():
        found = list(folder.rglob("*.pt"))
        if not found:
            raise FileNotFoundError(f"No .pt in {folder}")
        return found[0]
    return pt


def _load_piece_model(pt_path: Path) -> nn.Module:
    from chesscog.piece_classifier.models import InceptionV3  # noqa: PLC0415

    model = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(model, InceptionV3):
        model.eval()
        return _InceptionExportWrapper(model)
    if isinstance(model, dict) and "state_dict" in model:
        net = InceptionV3()
        net.load_state_dict(model["state_dict"])
        net.eval()
        return _InceptionExportWrapper(net)
    raise TypeError(f"Unexpected piece checkpoint type: {type(model)}")


def _load_occupancy_model(pt_path: Path) -> nn.Module:
    from chesscog.occupancy_classifier.models import ResNet  # noqa: PLC0415

    model = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(model, ResNet):
        model.eval()
        return model
    if isinstance(model, dict) and "state_dict" in model:
        net = ResNet()
        net.load_state_dict(model["state_dict"])
        net.eval()
        return net
    raise TypeError(f"Unexpected occupancy checkpoint type: {type(model)}")


class _InceptionExportWrapper(nn.Module):
    """Export wrapper — inference mode returns primary logits tensor."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inner.eval()
        out = self.inner(x)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "logits"):
            return out.logits
        return out


def _export_onnx(model: nn.Module, path: Path, *, image_size: int, num_classes: int, source: str) -> None:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["square"],
        output_names=["logits"],
        dynamic_axes={"square": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {path} ({image_size}x{image_size}, {num_classes} classes)")


def _write_piece_metadata(onnx_path: Path) -> None:
    meta = {
        "source": "chesscog",
        "model": "InceptionV3",
        "image_size": 299,
        "num_classes": 12,
        "class_names": list(CHESSCOG_PIECE_CLASSES),
        "visionchess_class_names": list(VISIONCHESS_PIECE_CLASSES),
        "class_map": _chesscog_to_visionchess_map(),
        "includes_empty_class": False,
        "input_name": "square",
        "output_name": "logits",
        "preprocess": "imagenet_normalize_rgb",
        "notes": "Use with chesscog occupancy; empty squares from occupancy model only.",
    }
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _write_occ_metadata(onnx_path: Path) -> None:
    meta = {
        "source": "chesscog",
        "model": "ResNet18",
        "image_size": 100,
        "num_classes": 2,
        "class_names": ["empty", "occupied"],
        "input_name": "square",
        "output_name": "logits",
        "preprocess": "imagenet_normalize_rgb",
    }
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _chesscog_to_visionchess_map() -> dict[str, str]:
    return {name: name for name in CHESSCOG_PIECE_CLASSES}


if __name__ == "__main__":
    main()
