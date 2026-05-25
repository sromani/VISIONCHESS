"""Soft probability calibration — no per-square hard thresholds."""

from __future__ import annotations

import numpy as np


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def fuse_soft_probability(
    fg: float,
    sil: float,
    edge: float,
    ent: float,
    center: float,
    ml_prob: float | None,
    *,
    weight_foreground: float = 0.32,
    weight_silhouette: float = 0.28,
    weight_edge: float = 0.18,
    weight_entropy: float = 0.12,
    weight_center: float = 0.10,
    ml_weight: float = 0.55,
    temperature: float = 1.15,
) -> float:
    """Blend signals in logit space → calibrated probability in (0, 1)."""
    heuristic = (
        weight_foreground * fg
        + weight_silhouette * sil
        + weight_edge * edge
        + weight_entropy * ent
        + weight_center * center
    )
    heuristic = float(np.clip(heuristic, 0.02, 0.98))

    if ml_prob is None:
        return sigmoid(logit(heuristic) / temperature)

    ml_prob = float(np.clip(ml_prob, 0.02, 0.98))
    blended_logit = (1.0 - ml_weight) * logit(heuristic) + ml_weight * logit(ml_prob)
    return sigmoid(blended_logit / temperature)


def adaptive_board_threshold(
    probabilities: list[float],
    *,
    target_pieces: int = 24,
    soft_floor: float = 0.16,
    soft_ceiling: float = 0.58,
) -> float:
    """Derive a board-level cutoff from the distribution — not a global constant."""
    if not probabilities:
        return soft_ceiling

    sorted_desc = sorted(probabilities, reverse=True)
    idx = min(max(target_pieces - 1, 0), len(sorted_desc) - 1)
    rank_cutoff = sorted_desc[idx]

    # Otsu-style gap on probability histogram (soft separation empty vs occupied cluster)
    gap_cutoff = _otsu_threshold(probabilities)

    cutoff = min(rank_cutoff, gap_cutoff) if gap_cutoff > soft_floor else rank_cutoff
    return float(np.clip(cutoff, soft_floor, soft_ceiling))


def assign_occupied_soft(
    probabilities: dict[str, float],
    *,
    target_pieces: int = 24,
    soft_floor: float = 0.16,
    soft_ceiling: float = 0.58,
    min_pieces: int = 0,
    max_pieces: int = 36,
) -> tuple[dict[str, bool], float]:
    """Assign occupied flags from soft probabilities using board-level calibration."""
    values = list(probabilities.values())
    threshold = adaptive_board_threshold(
        values,
        target_pieces=target_pieces,
        soft_floor=soft_floor,
        soft_ceiling=soft_ceiling,
    )

    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    occupied: dict[str, bool] = {name: prob >= threshold for name, prob in probabilities.items()}

    count = sum(occupied.values())

    # Promote highest-probability squares if board looks too empty
    if count < min_pieces:
        for name, prob in ranked:
            if count >= min_pieces:
                break
            if not occupied[name] and prob >= soft_floor * 0.85:
                occupied[name] = True
                count += 1

    # Demote lowest if still over max (board prior backup)
    if count > max_pieces:
        for name, prob in reversed(ranked):
            if count <= max_pieces:
                break
            if occupied[name]:
                occupied[name] = False
                count -= 1

    return occupied, threshold


def _otsu_threshold(values: list[float], bins: int = 16) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.5
    hist, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.5

    best_t = 0.5
    best_var = -1.0
    for i in range(1, bins):
        w0 = hist[:i].sum() / total
        w1 = hist[i:].sum() / total
        if w0 <= 0 or w1 <= 0:
            continue
        mu0 = np.average(centers[:i], weights=hist[:i] + 1e-8)
        mu1 = np.average(centers[i:], weights=hist[i:] + 1e-8)
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = float((edges[i - 1] + edges[i]) / 2.0)
    return best_t
