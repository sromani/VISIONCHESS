"""Generate synthetic training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset.synthetic_board import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic square dataset")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "squares")
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(
        args.root,
        samples_per_class=args.samples_per_class,
        size=args.size,
        seed=args.seed,
    )
    print(f"Dataset written to {args.root}")


if __name__ == "__main__":
    main()
