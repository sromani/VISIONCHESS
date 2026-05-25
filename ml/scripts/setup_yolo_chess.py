#!/usr/bin/env python3
"""Download pretrained chess piece YOLO detectors."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODELS = {
    "nakst": {
        "repo": "NAKSTStudio/yolov8m-chess-piece-detection",
        "file": "best.onnx",
        "dest": "yolov8_chess_pieces.onnx",
        "description": "YOLOv8m · 50k+ real/screenshot images · 640px normalized coords",
    },
    "yolo11n": {
        "repo": "yamero999/chess-piece-detection-yolo11n",
        "file": "best_mobile.onnx",
        "dest": "yolo11n_chess_pieces.onnx",
        "description": "YOLO11n · diverse boards/lighting · 416px",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download chess piece YOLO ONNX models")
    parser.add_argument(
        "--model",
        choices=["nakst", "yolo11n", "all"],
        default="all",
        help="Which model to download (default: all)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "ml" / "models" / "pretrained"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    for key in keys:
        spec = MODELS[key]
        print(f"Downloading {key}: {spec['description']}")
        src = hf_hub_download(spec["repo"], spec["file"])
        dest = out_dir / spec["dest"]
        shutil.copy2(src, dest)
        print(f"  -> {dest} ({dest.stat().st_size // 1024} KB)")

    print("Done. Set VISIONCHESS_YOLO_MODEL=nakst|yolo11n or path to .onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
