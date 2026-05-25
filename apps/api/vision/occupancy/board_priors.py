"""Board-level refinement after soft probability assignment."""

from __future__ import annotations

from dataclasses import replace

from vision.occupancy.config import OccupancyConfig
from vision.occupancy.types import OccupancyResult


def refine_board_selection(
    results: dict[str, OccupancyResult],
    config: OccupancyConfig,
) -> tuple[dict[str, OccupancyResult], bool]:
    """Trim excess occupied squares — never promote via hard threshold here."""
    occupied_names = [name for name, r in results.items() if r.occupied]
    count = len(occupied_names)

    if count <= config.max_expected_pieces:
        return results, False

    target = config.max_expected_pieces
    if count > config.hard_max_pieces:
        target = config.hard_max_pieces

    ranked = sorted(occupied_names, key=lambda n: results[n].probability)
    to_demote = count - target
    out = dict(results)
    for name in ranked[:to_demote]:
        prev = out[name]
        out[name] = replace(
            prev,
            occupied=False,
            score=min(config.confidence_empty, 0.50 + (1.0 - prev.probability) * 0.44),
            reason="board_prior_excess",
        )

    return out, True
