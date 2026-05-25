"""Build balanced binary occupancy dataset: empty/ vs occupied/."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect(root: Path, label: str) -> list[Path]:
    folder = root / label
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def build_binary_dataset(
    source: Path,
    dest: Path,
    *,
    max_per_class: int | None = None,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    piece_classes = [
        d.name
        for d in sorted((source / "train").iterdir())
        if d.is_dir() and d.name != "empty"
    ]

    empty_train = _collect(source / "train", "empty")
    empty_val = _collect(source / "val", "empty")
    occupied_train: list[Path] = []
    occupied_val: list[Path] = []
    for name in piece_classes:
        occupied_train.extend(_collect(source / "train", name))
        occupied_val.extend(_collect(source / "val", name))

    rng.shuffle(empty_train)
    rng.shuffle(empty_val)
    rng.shuffle(occupied_train)
    rng.shuffle(occupied_val)

    cap = max_per_class
    if cap is None:
        cap = min(len(empty_train), len(occupied_train))
    cap = max(cap, 100)

    empty_train = empty_train[:cap]
    empty_val = empty_val[: max(cap // 5, 20)]
    occupied_train = occupied_train[:cap]
    occupied_val = occupied_val[: max(cap // 5, 20)]

    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "train").exists():
        shutil.rmtree(dest)

    for split, empty_files, occ_files in (
        ("train", empty_train, occupied_train),
        ("val", empty_val, occupied_val),
    ):
        empty_dir = dest / split / "empty"
        occ_dir = dest / split / "occupied"
        empty_dir.mkdir(parents=True, exist_ok=True)
        occ_dir.mkdir(parents=True, exist_ok=True)

        for i, path in enumerate(empty_files):
            shutil.copy2(path, empty_dir / f"empty_{i:05d}{path.suffix}")
        for i, path in enumerate(occ_files):
            shutil.copy2(path, occ_dir / f"occupied_{i:05d}{path.suffix}")

    print(f"Built {dest}")
    print(f"  train: empty={len(empty_train)} occupied={len(occupied_train)}")
    print(f"  val:   empty={len(empty_val)} occupied={len(occupied_val)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/squares"))
    parser.add_argument("--dest", type=Path, default=Path("data/occupancy"))
    parser.add_argument("--max-per-class", type=int, default=400)
    args = parser.parse_args()
    build_binary_dataset(args.source, args.dest, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
