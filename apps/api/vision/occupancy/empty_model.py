"""Empty square background model — separate light and dark templates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision.occupancy.features import SquareFeatures, feature_vector, is_light_square


@dataclass
class EmptySquareTemplate:
    """Statistical model of an empty square for one board color."""

    mean_vector: NDArray[np.float32]
    std_vector: NDArray[np.float32]
    mean_lab: tuple[float, float, float]
    sample_count: int


@dataclass
class EmptySquareModel:
    light: EmptySquareTemplate
    dark: EmptySquareTemplate

    def template_for(self, is_light: bool) -> EmptySquareTemplate:
        return self.light if is_light else self.dark


def build_empty_model(
    squares: list[tuple[str, int, int, SquareFeatures, float]],
    *,
    seed_fraction: float = 0.45,
) -> EmptySquareModel:
    """Build light/dark empty templates from lowest-activation squares on this board."""
    light = _build_color_template(squares, is_light=True, seed_fraction=seed_fraction)
    dark = _build_color_template(squares, is_light=False, seed_fraction=seed_fraction)
    return EmptySquareModel(light=light, dark=dark)


def foreground_deviation(features: SquareFeatures, template: EmptySquareTemplate) -> float:
    """Normalized distance from empty template — higher means more likely occupied."""
    vec = feature_vector(features)
    std = np.maximum(template.std_vector, 1e-3)
    z = np.abs(vec - template.mean_vector) / std
    # Emphasize center-relevant dimensions (histogram + local variance + edges)
    weights = np.ones_like(z, dtype=np.float32)
    weights[3:19] = 1.4  # hist_l
    weights[-4:-1] = 1.6  # local vars + texture-related tail
    score = float(np.mean(z * weights))
    return min(score / 4.0, 1.0)


def _build_color_template(
    squares: list[tuple[str, int, int, SquareFeatures, float]],
    *,
    is_light: bool,
    seed_fraction: float,
) -> EmptySquareTemplate:
    group = [(name, row, col, feat, raw) for name, row, col, feat, raw in squares if is_light_square(row, col) == is_light]
    if not group:
        return _default_template()

    group.sort(key=lambda item: item[4])
    seed_n = max(2, int(len(group) * seed_fraction))
    seeds = group[:seed_n]
    vectors = np.stack([feature_vector(item[3]) for item in seeds], axis=0)
    mean_lab = tuple(float(np.mean([item[3].mean_lab[i] for item in seeds])) for i in range(3))

    return EmptySquareTemplate(
        mean_vector=vectors.mean(axis=0).astype(np.float32),
        std_vector=(vectors.std(axis=0) + 0.15).astype(np.float32),
        mean_lab=mean_lab,  # type: ignore[return-value]
        sample_count=len(seeds),
    )


def _default_template() -> EmptySquareTemplate:
    vec = np.zeros(28, dtype=np.float32)
    return EmptySquareTemplate(
        mean_vector=vec,
        std_vector=np.ones(28, dtype=np.float32),
        mean_lab=(128.0, 128.0, 128.0),
        sample_count=0,
    )
