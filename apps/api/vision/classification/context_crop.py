"""Context-aware square crops — expanded neighborhood centered on each cell."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.types import SQUARES_PER_SIDE, BoardGridResult, SquareCrop
from vision.classification.square_quality import DatasetSquareConfig, enhance_square_crop


@dataclass(frozen=True, slots=True)
class ContextCropConfig:
    """Extract N×N cell context windows centered on each square."""

    enabled: bool = True
    scale: float = 1.5  # 1.5 = 1.5×1.5 cells, 2.0 = 2×2 cells
    min_scale: float = 1.5
    max_scale: float = 2.0


def extract_context_crop(
    board_bgr: NDArray[np.uint8],
    row: int,
    col: int,
    *,
    scale: float = 1.5,
) -> NDArray[np.uint8]:
    """Axis-aligned context window on rectified board, clipped to edges."""
    if board_bgr.size == 0:
        return board_bgr

    h, w = board_bgr.shape[:2]
    cell = h // SQUARES_PER_SIDE
    cy = row * cell + cell // 2
    cx = col * cell + cell // 2
    half = max(cell // 2, int(round(cell * scale / 2.0)))

    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    crop = board_bgr[y0:y1, x0:x1]
    return _ensure_square_bgr(crop)


def build_context_grid(
    board_bgr: NDArray[np.uint8],
    reference: BoardGridResult,
    *,
    scale: float = 1.5,
    enhance_config: DatasetSquareConfig | None = None,
) -> BoardGridResult:
    """Build 64 context SquareCrops aligned with ``reference`` grid metadata."""
    cfg = enhance_config or DatasetSquareConfig()
    rows: list[tuple[SquareCrop, ...]] = []
    sizes: list[int] = []

    for row in reference.squares:
        cells: list[SquareCrop] = []
        for sq in row:
            raw = extract_context_crop(board_bgr, sq.row, sq.col, scale=scale)
            enhanced = enhance_square_crop(raw, cfg)
            sizes.append(enhanced.shape[0])
            cells.append(
                SquareCrop(
                    row=sq.row,
                    col=sq.col,
                    image=enhanced,
                    bbox=sq.bbox,
                    cell_bbox=sq.cell_bbox,
                    cell_quad=sq.cell_quad,
                    crop_quad=sq.crop_quad,
                )
            )
        rows.append(tuple(cells))

    avg_size = int(round(float(np.mean(sizes)))) if sizes else reference.square_size
    return BoardGridResult(
        squares=tuple(rows),
        board_size=reference.board_size,
        square_size=avg_size,
        margin_px=reference.margin_px,
        x_lines=reference.x_lines,
        y_lines=reference.y_lines,
        grid_method=f"{reference.grid_method}_context_{scale}x",
    )


def compare_crop_predictions(
    tight_by_name: dict[str, dict],
    context_by_name: dict[str, dict],
) -> dict:
    """A/B summary: tight 1×1 vs expanded context crops."""
    disagreements: list[dict] = []
    context_higher_conf: list[str] = []
    tight_higher_conf: list[str] = []
    large_piece_labels = {"white_queen", "black_queen", "white_rook", "black_rook", "white_bishop", "black_bishop"}

    for name in sorted(tight_by_name):
        tight = tight_by_name[name]
        context = context_by_name.get(name, {})
        t_label = str(tight.get("label", "unknown"))
        t_conf = float(tight.get("confidence", 0.0))
        c_label = str(context.get("label", "unknown"))
        c_conf = float(context.get("confidence", 0.0))

        if t_label != c_label:
            disagreements.append(
                {
                    "square_name": name,
                    "tight_label": t_label,
                    "tight_confidence": t_conf,
                    "context_label": c_label,
                    "context_confidence": c_conf,
                    "involves_large_piece": t_label in large_piece_labels or c_label in large_piece_labels,
                }
            )
        if c_conf > t_conf + 0.05:
            context_higher_conf.append(name)
        elif t_conf > c_conf + 0.05:
            tight_higher_conf.append(name)

    large_piece_disagreements = [d for d in disagreements if d["involves_large_piece"]]
    return {
        "tight": {"strategy": "1x1_tight_crop", "squares": len(tight_by_name)},
        "context": {"strategy": "expanded_context_crop", "squares": len(context_by_name)},
        "agreement_count": len(tight_by_name) - len(disagreements),
        "disagreement_count": len(disagreements),
        "context_higher_confidence": context_higher_conf,
        "tight_higher_confidence": tight_higher_conf,
        "large_piece_disagreements": large_piece_disagreements,
        "disagreements": disagreements[:32],
    }


def _ensure_square_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    if h == w:
        return image.copy()
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side].copy()
