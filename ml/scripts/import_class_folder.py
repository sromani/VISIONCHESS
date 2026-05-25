"""Import external class-folder datasets into VisionChess layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from training.labels import CLASS_NAMES, validate_class_name

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "data" / "squares" / "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import class-folder dataset")
    parser.add_argument("source", type=Path, help="Root with subdirs per class name")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 = unlimited")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")

    out_root = args.out if args.split == "train" else args.out.parent / "val"
    copied = 0
    for class_dir in sorted(args.source.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.lower().replace(" ", "_")
        try:
            validate_class_name(label)
        except ValueError:
            mapped = _map_external_label(label)
            if mapped is None:
                print(f"Skip unknown class: {class_dir.name}")
                continue
            label = mapped

        dest_dir = out_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        images = [
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if args.max_per_class > 0:
            images = images[: args.max_per_class]
        for img in images:
            target = dest_dir / f"{args.source.name}_{img.name}"
            shutil.copy2(img, target)
            copied += 1
        print(f"{label}: {len(images)} images")

    print(f"Imported {copied} images -> {out_root}")


def _map_external_label(name: str) -> str | None:
    """Map common external label conventions."""
    fen_map = {
        "p": "white_pawn", "n": "white_knight", "b": "white_bishop",
        "r": "white_rook", "q": "white_queen", "k": "white_king",
        "P": "white_pawn", "N": "white_knight", "B": "white_bishop",
        "R": "white_rook", "Q": "white_queen", "K": "white_king",
    }
    if name in fen_map:
        return fen_map[name]
    if name.startswith("w_"):
        return "white_" + name[2:]
    if name.startswith("b_"):
        return "black_" + name[2:]
    if name in CLASS_NAMES:
        return name
    return None


if __name__ == "__main__":
    main()
