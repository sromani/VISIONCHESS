"""Build dataset from Lichess puzzle FENs."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset.lichess_auto_label import fetch_lichess_puzzles, render_fen_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "squares")
    parser.add_argument("--puzzles", type=int, default=80)
    args = parser.parse_args()
    fens = fetch_lichess_puzzles(args.puzzles)
    n = render_fen_dataset(fens, args.root)
    print(f"Wrote {n} squares from {len(fens)} puzzles -> {args.root}")


if __name__ == "__main__":
    main()
