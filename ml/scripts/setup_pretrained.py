#!/usr/bin/env python3
"""Download or build pretrained piece + occupancy ONNX models.

Sources:
  synthetic  — train MobileNet on local synthetic data (fast bootstrap)
  lichess    — build dataset from Lichess puzzles then train
  chesscog   — export chesscog pretrained weights to ONNX (pip install chesscog)
  lc2fen     — copy LiveChess2FEN ONNX from --lc2fen-dir
  import-onnx — copy user ONNX + optional metadata JSON

Usage:
  python scripts/setup_pretrained.py --source synthetic --epochs 15
  python scripts/setup_pretrained.py --source chesscog
  python scripts/setup_pretrained.py --source lc2fen --lc2fen-dir ../LiveChess2FEN/data/models
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "ml" / "models"
PRETRAINED = MODELS / "pretrained"
PIECE_OUT = MODELS / "piece_classifier"


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup pretrained ML models")
    parser.add_argument(
        "--source",
        choices=["synthetic", "lichess", "chesscog", "lc2fen", "import-onnx"],
        default="synthetic",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lc2fen-dir", type=Path, default=None)
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--meta-path", type=Path, default=None)
    args = parser.parse_args()

    PRETRAINED.mkdir(parents=True, exist_ok=True)
    PIECE_OUT.mkdir(parents=True, exist_ok=True)

    if args.source == "synthetic":
        _train_local(epochs=args.epochs, puzzles=0)
    elif args.source == "lichess":
        _train_local(epochs=args.epochs, puzzles=120)
    elif args.source == "chesscog":
        _export_chesscog()
    elif args.source == "lc2fen":
        _import_lc2fen(args.lc2fen_dir)
    elif args.source == "import-onnx":
        _import_onnx(args.onnx_path, args.meta_path)

    print(f"Piece models dir: {PIECE_OUT}")
    print(f"Pretrained dir:   {PRETRAINED}")


def _train_local(*, epochs: int, puzzles: int) -> None:
    from scripts.generate_synthetic_dataset import main as gen_main  # noqa: PLC0415

    if puzzles > 0:
        from dataset.lichess_auto_label import fetch_lichess_puzzles, render_fen_dataset

        data_root = REPO / "ml" / "data" / "squares"
        fens = fetch_lichess_puzzles(puzzles)
        render_fen_dataset(fens, data_root)
        print(f"Lichess dataset: {puzzles} puzzles")
    else:
        import sys

        from dataset.synthetic_board import generate_dataset

        generate_dataset(REPO / "ml" / "data" / "squares", samples_per_class=200, size=64, seed=42)
        print("Synthetic dataset ready")

    from training.config import TrainConfig
    from training.cli import main as train_main
    import sys

    sys.argv = [
        "train-pieces",
        "--epochs",
        str(epochs),
        "--output-dir",
        str(PIECE_OUT),
    ]
    train_main()
    src = PIECE_OUT / "piece_classifier.onnx"
    if src.exists():
        shutil.copy2(src, PRETRAINED / "visionchess_piece.onnx")
        print(f"Installed -> {PRETRAINED / 'visionchess_piece.onnx'}")


def _export_chesscog() -> None:
    try:
        from scripts.export_chesscog_onnx import export_all
    except ImportError as exc:
        msg = "Install chesscog: pip install chesscog && python -m chesscog.piece_classifier.download_model"
        raise SystemExit(msg) from exc
    export_all(PRETRAINED)


def _import_lc2fen(lc2fen_dir: Path | None) -> None:
    if lc2fen_dir is None or not lc2fen_dir.exists():
        raise SystemExit(
            "Provide --lc2fen-dir pointing to LiveChess2FEN/data/models "
            "(download .onnx from GitHub releases)"
        )
    onnx_files = list(lc2fen_dir.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"No .onnx in {lc2fen_dir}")
    src = max(onnx_files, key=lambda p: p.stat().st_size)
    dest = PRETRAINED / "lc2fen_piece.onnx"
    shutil.copy2(src, dest)
    meta = {
        "source": "lc2fen",
        "image_size": 64,
        "num_classes": 13,
        "class_names": list(_default_classes()),
        "input_name": "input",
        "output_name": "output",
        "notes": "Verify input/output names match your LC2FEN export; edit JSON if needed.",
    }
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.copy2(dest, PIECE_OUT / "piece_classifier.onnx")
    shutil.copy2(dest.with_suffix(".json"), PIECE_OUT / "piece_classifier.json")
    print(f"Imported LC2FEN -> {dest}")


def _import_onnx(onnx_path: Path | None, meta_path: Path | None) -> None:
    if onnx_path is None or not onnx_path.exists():
        raise SystemExit("--onnx-path required")
    dest = PRETRAINED / "custom_piece.onnx"
    shutil.copy2(onnx_path, dest)
    if meta_path and meta_path.exists():
        shutil.copy2(meta_path, dest.with_suffix(".json"))
    else:
        meta = {
            "source": "custom",
            "image_size": 64,
            "num_classes": 13,
            "class_names": list(_default_classes()),
            "input_name": "square",
            "output_name": "logits",
        }
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.copy2(dest, PIECE_OUT / "piece_classifier.onnx")
    shutil.copy2(dest.with_suffix(".json"), PIECE_OUT / "piece_classifier.json")
    print(f"Imported custom ONNX -> {dest}")


def _default_classes() -> tuple[str, ...]:
    return (
        "empty",
        "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
        "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
    )


if __name__ == "__main__":
    main()
