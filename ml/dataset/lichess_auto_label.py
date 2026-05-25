"""Fetch Lichess puzzles and render labeled square crops."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import chess
import cv2
import numpy as np

from dataset.synthetic_board import render_square, square_label_from_board


def fetch_lichess_puzzles(count: int = 100) -> list[str]:
    """Download puzzle FENs from Lichess API."""
    fens: list[str] = []
    url = "https://lichess.org/api/puzzle/batch/50"
    while len(fens) < count:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for puzzle in data.get("puzzles", []):
            fen = puzzle.get("game", {}).get("initialFen") or puzzle.get("fen")
            if fen:
                fens.append(fen)
            if len(fens) >= count:
                break
    return fens[:count]


def render_fen_dataset(
    fens: list[str],
    root: Path,
    *,
    size: int = 64,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> int:
    rng = np.random.default_rng(seed)
    written = 0
    for fen in fens:
        board = chess.Board(fen)
        for sq in chess.SQUARES:
            label = square_label_from_board(board, sq)
            split = "val" if rng.random() < val_ratio else "train"
            out_dir = root / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            img = render_square(board, sq, size=size, rng=rng)
            path = out_dir / f"lichess_{written:06d}_{chess.square_name(sq)}.png"
            cv2.imwrite(str(path), img)
            written += 1
    return written
