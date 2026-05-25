"""Rich synthetic square generator for ML training."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from training.labels import CLASS_NAMES, FEN_SYMBOL_TO_LABEL


@dataclass(frozen=True, slots=True)
class BoardTheme:
    name: str
    light: tuple[int, int, int]
    dark: tuple[int, int, int]
    white_piece: tuple[int, int, int]
    black_piece: tuple[int, int, int]


THEMES: tuple[BoardTheme, ...] = (
    BoardTheme("classic", (240, 217, 181), (181, 136, 99), (255, 255, 255), (20, 20, 20)),
    BoardTheme("green", (238, 238, 210), (118, 150, 86), (250, 250, 250), (30, 30, 30)),
    BoardTheme("blue", (222, 227, 230), (140, 162, 173), (255, 255, 255), (15, 15, 15)),
    BoardTheme("dark", (120, 120, 120), (60, 60, 60), (230, 230, 230), (10, 10, 10)),
)


def random_board() -> chess.Board:
    board = chess.Board(chess.STARTING_FEN)
    for _ in range(np.random.randint(4, 28)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(np.random.choice(moves))
    return board


def square_label_from_board(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    if piece is None:
        return "empty"
    sym = piece.symbol()
    return FEN_SYMBOL_TO_LABEL[sym]


def render_square(
    board: chess.Board,
    square: chess.Square,
    *,
    size: int = 64,
    theme: BoardTheme | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.uint8]:
    """Render one square crop with optional piece silhouette."""
    rng = rng or np.random.default_rng()
    theme = theme or THEMES[int(rng.integers(0, len(THEMES)))]
    row, col = divmod(square, 8)
    is_light = (row + col) % 2 == 0
    bg = theme.light if is_light else theme.dark

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    piece = board.piece_at(square)
    if piece is not None:
        color = theme.white_piece if piece.color == chess.WHITE else theme.black_piece
        _draw_piece(draw, size, piece.symbol().lower(), color)

    arr = np.array(img)
    arr = _apply_degradations(arr, rng)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _draw_piece(draw: ImageDraw.ImageDraw, size: int, sym: str, color: tuple[int, int, int]) -> None:
    cx, cy = size // 2, size // 2
    r = size // 3
    if sym in {"p"}:
        draw.ellipse((cx - r // 2, cy - r, cx + r // 2, cy + r // 3), fill=color)
        draw.rectangle((cx - r // 3, cy, cx + r // 3, cy + r), fill=color)
    elif sym in {"n"}:
        draw.polygon([(cx, cy - r), (cx + r, cy + r // 2), (cx - r // 3, cy + r)], fill=color)
    elif sym in {"b"}:
        draw.polygon([(cx, cy - r), (cx + r // 2, cy + r), (cx - r // 2, cy + r)], fill=color)
        draw.ellipse((cx - r // 4, cy - r // 2, cx + r // 4, cy), fill=color)
    elif sym in {"r"}:
        draw.rectangle((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), fill=color)
    elif sym in {"q"}:
        draw.ellipse((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), fill=color)
        draw.polygon([(cx - r // 3, cy - r), (cx + r // 3, cy - r), (cx, cy - r // 2)], fill=color)
    elif sym in {"k"}:
        draw.rectangle((cx - r // 3, cy - r // 2, cx + r // 3, cy + r // 2), fill=color)
        draw.rectangle((cx - r // 6, cy - r // 2 - 4, cx + r // 6, cy - r // 2), fill=color)
        draw.rectangle((cx - 2, cy - r // 2 - 8, cx + 2, cy - r // 2 - 4), fill=color)


def _apply_degradations(arr: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    out = arr.copy()
    if rng.random() < 0.7:
        alpha = rng.uniform(0.75, 1.25)
        beta = rng.integers(-25, 26)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.2, 1.4))
    if rng.random() < 0.25:
        quality = int(rng.integers(35, 90))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    if rng.random() < 0.3:
        angle = rng.uniform(-12, 12)
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, rng.uniform(0.92, 1.08))
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    noise = rng.integers(-10, 11, size=out.shape, dtype=np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def generate_dataset(
    root: Path,
    *,
    samples_per_class: int = 200,
    val_ratio: float = 0.15,
    size: int = 64,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for name in CLASS_NAMES:
            (root / split / name).mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in CLASS_NAMES}
    target = samples_per_class * len(CLASS_NAMES)

    while sum(counts.values()) < target:
        board = random_board()
        theme = THEMES[int(rng.integers(0, len(THEMES)))]
        for sq in chess.SQUARES:
            label = square_label_from_board(board, sq)
            if counts[label] >= samples_per_class:
                continue
            img = render_square(board, sq, size=size, theme=theme, rng=rng)
            split = "val" if rng.random() < val_ratio else "train"
            out = root / split / label / f"{label}_{counts[label]:05d}.png"
            cv2.imwrite(str(out), img)
            counts[label] += 1
            if sum(counts.values()) >= target:
                break
