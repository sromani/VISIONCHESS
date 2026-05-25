#!/usr/bin/env python3
"""Download LiveChess2FEN ONNX models into ml/models/lc2fen."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "ml" / "vendor" / "LiveChess2FEN"
MODELS_DIR = REPO_ROOT / "ml" / "models" / "lc2fen"
LAPS_DIR = VENDOR_ROOT / "lc2fen" / "detectboard" / "models"

PIECE_URL = (
    "https://github.com/davidmallasen/LiveChess2FEN/releases/download/v0.1.0/"
    "MobileNetV2_0p5_all.onnx"
)
LAPS_ONNX_URL = (
    "https://raw.githubusercontent.com/davidmallasen/LiveChess2FEN/master/"
    "lc2fen/detectboard/models/laps_model.onnx"
)
LAPS_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/davidmallasen/LiveChess2FEN/master/"
    "lc2fen/detectboard/models/laps.weights.h5"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip (exists): {dest}")
        return
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup LiveChess2FEN models")
    parser.add_argument(
        "--clone",
        action="store_true",
        help="Clone vendor repo if missing (requires git)",
    )
    args = parser.parse_args()

    if not VENDOR_ROOT.is_dir():
        if not args.clone:
            print(
                f"Vendor missing at {VENDOR_ROOT}. "
                "Clone with: git clone --depth 1 "
                "https://github.com/davidmallasen/LiveChess2FEN.git "
                f"{VENDOR_ROOT}",
                file=sys.stderr,
            )
            return 1
        import subprocess

        VENDOR_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/davidmallasen/LiveChess2FEN.git",
                str(VENDOR_ROOT),
            ]
        )

    _download(PIECE_URL, MODELS_DIR / "MobileNetV2_0p5_all.onnx")
    _download(LAPS_ONNX_URL, LAPS_DIR / "laps_model.onnx")
    _download(LAPS_WEIGHTS_URL, LAPS_DIR / "laps.weights.h5")

    pretrained = REPO_ROOT / "ml" / "models" / "pretrained" / "lc2fen_piece.onnx"
    piece_model = MODELS_DIR / "MobileNetV2_0p5_all.onnx"
    if piece_model.exists():
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(piece_model, pretrained)
        print(f"linked pretrained copy -> {pretrained}")

    print("LiveChess2FEN models ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
