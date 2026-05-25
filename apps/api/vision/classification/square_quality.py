"""Convert perspective-warped square crops into dataset-quality inputs.

Training data uses fixed-size, centered, lighting-normalized square images.
Raw warps still carry border bleed and exposure variance — this module aligns
inference crops with what the classifier was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.types import BoardGridResult, SquareCrop


@dataclass(frozen=True, slots=True)
class DatasetSquareConfig:
    """Parameters matching training / Chessvision-style square crops."""

    output_size: int = 64
    inner_crop_ratio: float = 0.90
    clahe_clip: float = 2.0
    clahe_grid: int = 4
    bilateral_d: int = 5
    bilateral_sigma: float = 50.0
    sharpen_amount: float = 0.35
    max_border_shave_ratio: float = 0.06
    min_analysis_px: int = 128
    # Context crop experiment (does not replace tight crops in fusion)
    context_crop_enabled: bool = True
    context_crop_scale: float = 1.5


def enhance_square_crop(
    crop_bgr: NDArray[np.uint8],
    config: DatasetSquareConfig | None = None,
) -> NDArray[np.uint8]:
    """Enhance a square crop at native resolution — no downscale."""
    cfg = config or DatasetSquareConfig()
    if crop_bgr.size == 0:
        return crop_bgr

    square = _ensure_square_bgr(crop_bgr)
    square = _shave_border_bleed(square, cfg)
    square = _normalize_lighting(square, _adaptive_clahe_grid(square.shape[0], cfg), clip=cfg.clahe_clip)
    square = _center_focus(square, cfg.inner_crop_ratio)
    square = cv2.bilateralFilter(square, cfg.bilateral_d, cfg.bilateral_sigma, cfg.bilateral_sigma)
    return _unsharp_mask(square, cfg.sharpen_amount)


def enhance_grid_for_analysis(
    grid: BoardGridResult,
    config: DatasetSquareConfig | None = None,
) -> BoardGridResult:
    """Maximize per-square visual quality while preserving extraction resolution."""
    cfg = config or DatasetSquareConfig()
    rows: list[tuple[SquareCrop, ...]] = []
    sizes: list[int] = []

    for row in grid.squares:
        cells: list[SquareCrop] = []
        for sq in row:
            enhanced = enhance_square_crop(sq.image, cfg)
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

    avg_size = int(round(float(np.mean(sizes)))) if sizes else grid.square_size
    return BoardGridResult(
        squares=tuple(rows),
        board_size=grid.board_size,
        square_size=avg_size,
        margin_px=grid.margin_px,
        x_lines=grid.x_lines,
        y_lines=grid.y_lines,
        grid_method=f"{grid.grid_method}_enhanced",
    )


def normalize_square_for_dataset(
    crop_bgr: NDArray[np.uint8],
    config: DatasetSquareConfig | None = None,
) -> NDArray[np.uint8]:
    """Enhanced crop → fixed-size export for dataset / UI preview."""
    cfg = config or DatasetSquareConfig()
    if crop_bgr.size == 0:
        return np.zeros((cfg.output_size, cfg.output_size, 3), dtype=np.uint8)

    square = enhance_square_crop(crop_bgr, cfg)
    if square.shape[0] != cfg.output_size or square.shape[1] != cfg.output_size:
        square = cv2.resize(square, (cfg.output_size, cfg.output_size), interpolation=cv2.INTER_AREA)
    return square


def normalize_grid_for_dataset(
    grid: BoardGridResult,
    config: DatasetSquareConfig | None = None,
) -> BoardGridResult:
    """Downscale enhanced analysis crops to fixed model/export size."""
    cfg = config or DatasetSquareConfig()
    rows: list[tuple[SquareCrop, ...]] = []

    for row in grid.squares:
        cells: list[SquareCrop] = []
        for sq in row:
            normalized = normalize_square_for_dataset(sq.image, cfg)
            cells.append(
                SquareCrop(
                    row=sq.row,
                    col=sq.col,
                    image=normalized,
                    bbox=sq.bbox,
                    cell_bbox=sq.cell_bbox,
                    cell_quad=sq.cell_quad,
                    crop_quad=sq.crop_quad,
                )
            )
        rows.append(tuple(cells))

    return BoardGridResult(
        squares=tuple(rows),
        board_size=grid.board_size,
        square_size=cfg.output_size,
        margin_px=grid.margin_px,
        x_lines=grid.x_lines,
        y_lines=grid.y_lines,
        grid_method="dataset_export",
    )


def render_dataset_montage(
    grid: BoardGridResult,
    cell_px: int = 80,
    gutter: int = 2,
) -> NDArray[np.uint8]:
    """8×8 montage of dataset-quality crops (classifier input preview)."""
    label_h = 16
    tile = cell_px + label_h
    canvas_size = 8 * tile + 9 * gutter
    canvas = np.full((canvas_size, canvas_size, 3), 28, dtype=np.uint8)

    for sq in grid.flat:
        resized = cv2.resize(sq.image, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)
        ox = gutter + sq.col * (tile + gutter)
        oy = gutter + sq.row * (tile + gutter)
        canvas[oy + label_h : oy + label_h + cell_px, ox : ox + cell_px] = resized
        cv2.putText(
            canvas,
            sq.square_name,
            (ox + 3, oy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    return canvas


def _ensure_square_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    if h == w:
        return image
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side].copy()


def _shave_border_bleed(square: NDArray[np.uint8], config: DatasetSquareConfig) -> NDArray[np.uint8]:
    """Remove 1–N px from edges when outer ring looks like neighbor contamination."""
    h, w = square.shape[:2]
    max_shave = max(1, int(min(h, w) * config.max_border_shave_ratio))
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    inner = gray[max_shave:-max_shave, max_shave:-max_shave]
    if inner.size == 0:
        return square

    inner_std = float(np.std(inner))
    shave = 0
    for px in range(1, max_shave + 1):
        ring = np.concatenate(
            [
                gray[:px, :].ravel(),
                gray[-px:, :].ravel(),
                gray[px:-px, :px].ravel(),
                gray[px:-px, -px:].ravel(),
            ]
        )
        if float(np.std(ring)) > inner_std * 1.35:
            shave = px
        else:
            break

    if shave <= 0:
        return square
    return square[shave : h - shave, shave : w - shave].copy()


def _normalize_lighting(square: NDArray[np.uint8], clahe_grid: int, *, clip: float = 2.0) -> NDArray[np.uint8]:
    lab = cv2.cvtColor(square, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(clahe_grid, clahe_grid),
    )
    l_norm = clahe.apply(l_channel)
    merged = cv2.merge([l_norm, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _adaptive_clahe_grid(side_px: int, config: DatasetSquareConfig) -> int:
    return max(config.clahe_grid, min(16, side_px // 24))


def _center_focus(square: NDArray[np.uint8], ratio: float) -> NDArray[np.uint8]:
    h, w = square.shape[:2]
    side = max(8, int(min(h, w) * ratio))
    cy, cx = h // 2, w // 2
    y0 = max(cy - side // 2, 0)
    x0 = max(cx - side // 2, 0)
    y1 = min(y0 + side, h)
    x1 = min(x0 + side, w)
    return square[y0:y1, x0:x1].copy()


def _unsharp_mask(image: NDArray[np.uint8], amount: float) -> NDArray[np.uint8]:
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
