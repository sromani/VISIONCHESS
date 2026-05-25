"""Multi-signal occupancy detector — probabilities only, no binary decisions."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.types import BoardGridResult
from vision.occupancy.calibration import adaptive_board_threshold, fuse_soft_probability
from vision.occupancy.config import OccupancyConfig
from vision.occupancy.empty_model import build_empty_model, foreground_deviation
from vision.occupancy.features import extract_features, is_light_square
from vision.occupancy.ml_model import MlOccupancyModel
from vision.inference.model_registry import resolve_occupancy_model
from vision.occupancy.silhouette import edge_score, entropy_score, silhouette_score
from vision.occupancy.types import OccupancyReport, OccupancyResult, SquareOccupancyDebug


class OccupancyDetector:
    def __init__(self, config: OccupancyConfig | None = None) -> None:
        self._config = config or OccupancyConfig()
        self._ml: MlOccupancyModel | None = None
        artifact = resolve_occupancy_model(self._config.ml_model_path)
        if artifact is not None:
            try:
                self._ml = MlOccupancyModel.from_artifact(artifact)
            except Exception:
                self._ml = None

    def detect_grid(
        self,
        grid: BoardGridResult,
        *,
        capture_ml_debug: bool = False,
    ) -> OccupancyReport:
        squares = list(grid.flat)
        cfg = self._config

        seed_data: list[tuple[str, int, int, object, float]] = []
        crops: dict[str, NDArray[np.uint8]] = {}
        for sq in squares:
            bgr = _ensure_bgr(sq.image)
            crops[sq.square_name] = bgr
            if not cfg.ml_only:
                feat = extract_features(bgr)
                sil, _ = silhouette_score(bgr)
                raw = _raw_activation(bgr, feat)
                seed_data.append((sq.square_name, sq.row, sq.col, feat, raw + sil * 0.6))

        empty_model = build_empty_model(seed_data, seed_fraction=cfg.empty_seed_fraction) if not cfg.ml_only else None

        probabilities: dict[str, float] = {}
        debug_rows: list[SquareOccupancyDebug] = []
        results: dict[str, OccupancyResult] = {}
        ml_debug: dict[str, object] = {}

        for sq in squares:
            bgr = crops[sq.square_name]
            ml_prob: float | None = None
            if self._ml is not None:
                if capture_ml_debug:
                    occ_dbg = self._ml.predict_debug(bgr)
                    ml_prob = occ_dbg.occupied_probability
                    ml_debug[sq.square_name] = occ_dbg
                else:
                    ml_prob = self._ml.predict(bgr)

            if cfg.ml_only:
                if self._ml is None:
                    msg = "ml_only occupancy requires occupancy.onnx — train with ml/training/occupancy_cli"
                    raise RuntimeError(msg)
                prob = float(ml_prob)
                fg = sil = edge = ent = 0.0
                center = 0.0
                reason = "ml_only"
            else:
                feat = extract_features(bgr)
                assert empty_model is not None
                template = empty_model.template_for(is_light_square(sq.row, sq.col))
                fg = foreground_deviation(feat, template)
                sil, center = silhouette_score(bgr)
                edge = edge_score(feat.edge_density_center, feat.edge_density_border)
                ent = entropy_score(feat.entropy)
                prob = fuse_soft_probability(
                    fg, sil, edge, ent, center, ml_prob,
                    weight_foreground=cfg.weight_foreground,
                    weight_silhouette=cfg.weight_silhouette,
                    weight_edge=cfg.weight_edge,
                    weight_entropy=cfg.weight_entropy,
                    weight_center=cfg.weight_center,
                    ml_weight=cfg.ml_weight,
                    temperature=cfg.calibration_temperature,
                )
                reason = "soft_pending"

            probabilities[sq.square_name] = prob
            debug_rows.append(
                SquareOccupancyDebug(
                    square_name=sq.square_name,
                    is_light=is_light_square(sq.row, sq.col),
                    foreground_score=fg,
                    silhouette_score=sil,
                    edge_score=edge,
                    entropy_score=ent,
                    center_activation=center,
                    ml_probability=ml_prob,
                    fused_probability=prob,
                    occupied=False,
                    reason=reason,
                )
            )
            results[sq.square_name] = OccupancyResult(
                occupied=False,
                score=prob,
                probability=prob,
                foreground_score=fg,
                silhouette_score=sil,
                edge_score=edge,
                entropy_score=ent,
                center_activation=center,
                reason=reason,
                debug=debug_rows[-1],
            )

        board_threshold = adaptive_board_threshold(
            list(probabilities.values()),
            target_pieces=cfg.target_pieces,
            soft_floor=cfg.soft_floor,
            soft_ceiling=cfg.soft_ceiling,
        )
        likely_count = sum(1 for p in probabilities.values() if p >= board_threshold)

        return OccupancyReport(
            results=results,
            occupied_count=likely_count,
            empty_count=64 - likely_count,
            board_prior_applied=False,
            board_threshold=board_threshold,
            debug_rows=tuple(debug_rows),
            ml_debug=ml_debug if ml_debug else None,
        )


def detect_board_occupancy(
    grid: BoardGridResult,
    config: OccupancyConfig | None = None,
) -> OccupancyReport:
    return OccupancyDetector(config).detect_grid(grid)


def detect_square(
    crop_bgr: NDArray[np.uint8],
    row: int,
    col: int,
    square_name: str = "a1",
    *,
    config: OccupancyConfig | None = None,
) -> OccupancyResult:
    """Single-square soft probability (no binary commit)."""
    cfg = config or OccupancyConfig()
    bgr = _ensure_bgr(crop_bgr)
    feat = extract_features(bgr)

    fg = _local_foreground(bgr, feat)
    sil, center = silhouette_score(bgr)
    edge = edge_score(feat.edge_density_center, feat.edge_density_border)
    ent = entropy_score(feat.entropy)

    prob = fuse_soft_probability(
        fg, sil, edge, ent, center, None,
        weight_foreground=cfg.weight_foreground,
        weight_silhouette=cfg.weight_silhouette,
        weight_edge=cfg.weight_edge,
        weight_entropy=cfg.weight_entropy,
        weight_center=cfg.weight_center,
        ml_weight=cfg.ml_weight,
        temperature=cfg.calibration_temperature,
    )

    return OccupancyResult(
        occupied=False,
        score=prob,
        probability=prob,
        foreground_score=fg,
        silhouette_score=sil,
        edge_score=edge,
        entropy_score=ent,
        center_activation=center,
        reason="soft_pending",
        debug=SquareOccupancyDebug(
            square_name=square_name,
            is_light=is_light_square(row, col),
            foreground_score=fg,
            silhouette_score=sil,
            edge_score=edge,
            entropy_score=ent,
            center_activation=center,
            ml_probability=None,
            fused_probability=prob,
            occupied=False,
            reason="soft_pending",
        ),
    )


def _local_foreground(bgr: NDArray[np.uint8], feat) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ring = max(2, int(min(h, w) * 0.14))
    border = np.concatenate(
        [
            gray[:ring, :].ravel(),
            gray[-ring:, :].ravel(),
            gray[ring:-ring, :ring].ravel(),
            gray[ring:-ring, -ring:].ravel(),
        ]
    )
    bg = float(np.median(border))
    center = gray[ring : h - ring, ring : w - ring]
    diff = float(np.mean(np.abs(center.astype(np.float32) - bg) > 20.0))
    texture = min(feat.texture_energy / 400.0, 1.0)
    return float(np.clip(0.55 * diff + 0.45 * texture, 0.0, 1.0))


def _raw_activation(bgr: NDArray[np.uint8], feat) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(gray.shape) / 5.0)
    hp = cv2.absdiff(gray, blur)
    h, w = gray.shape
    m = max(2, int(min(h, w) * 0.18))
    center = hp[m : h - m, m : w - m]
    return float(np.mean(center)) + feat.texture_energy * 0.002 + feat.edge_density_center * 0.5


def _ensure_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
