"""Final board binarization — only stage that commits occupied/empty."""

from __future__ import annotations

from dataclasses import replace

from vision.classification.chess_correction import apply_chess_constraints
from vision.classification.types import SquareClassification
from vision.occupancy.board_priors import refine_board_selection
from vision.occupancy.calibration import assign_occupied_soft
from vision.occupancy.config import OccupancyConfig
from vision.occupancy.types import OccupancyResult
from vision.scanner.context import ScanContext


def finalize_board(
    soft_squares: list[SquareClassification],
    occupancy: dict[str, OccupancyResult],
    config: OccupancyConfig,
) -> tuple[list[SquareClassification], float, bool]:
    """Convert soft probabilities into final occupied/empty labels."""
    probabilities = {name: occ.probability for name, occ in occupancy.items()}
    occupied_flags, board_threshold = assign_occupied_soft(
        probabilities,
        target_pieces=config.target_pieces,
        soft_floor=config.soft_floor,
        soft_ceiling=config.soft_ceiling,
        min_pieces=0,
        max_pieces=config.hard_max_pieces,
    )

    by_name = {sq.square_name: sq for sq in soft_squares}
    finalized: list[SquareClassification] = []

    for name, occ in occupancy.items():
        sq = by_name.get(name)
        if sq is None:
            continue

        occ_prob = occ.probability
        is_occupied = occupied_flags[name]

        if not is_occupied:
            empty_conf = min(config.confidence_empty, 0.50 + (1.0 - occ_prob) * 0.44)
            finalized.append(
                replace(
                    sq,
                    label="empty",
                    confidence=empty_conf,
                    occupied=False,
                    occupancy_score=occ_prob,
                    empty_reason="finalized_empty",
                )
            )
            continue

        piece_label = sq.label if sq.label != "empty" else "empty"
        piece_conf = sq.confidence
        joint_conf = float(occ_prob * piece_conf) if piece_label != "empty" else occ_prob * 0.5

        if piece_label == "empty" or joint_conf < config.soft_floor:
            finalized.append(
                replace(
                    sq,
                    label="empty",
                    confidence=min(config.confidence_empty, 0.50 + (1.0 - occ_prob) * 0.44),
                    occupied=False,
                    occupancy_score=occ_prob,
                    empty_reason="finalized_low_joint",
                )
            )
            continue

        finalized.append(
            replace(
                sq,
                label=piece_label,
                confidence=joint_conf,
                occupied=True,
                occupancy_score=occ_prob,
                empty_reason=None,
            )
        )

    finalized = apply_chess_constraints(finalized)

    occ_results = {
        name: replace(occ, occupied=occupied_flags[name], score=occ.probability)
        for name, occ in occupancy.items()
    }
    occ_results, prior_applied = refine_board_selection(occ_results, config)

    if prior_applied:
        demoted = {n for n, r in occ_results.items() if not r.occupied}
        out: list[SquareClassification] = []
        for sq in finalized:
            if sq.square_name in demoted and sq.occupied:
                occ_prob = occupancy[sq.square_name].probability
                out.append(
                    replace(
                        sq,
                        label="empty",
                        confidence=min(config.confidence_empty, 0.50 + (1.0 - occ_prob) * 0.44),
                        occupied=False,
                        occupancy_score=occ_prob,
                        empty_reason="board_prior_excess",
                    )
                )
            else:
                out.append(sq)
        finalized = out

    finalized.sort(key=lambda s: (s.row, s.col))
    return finalized, board_threshold, prior_applied


def finalize_from_context(ctx: ScanContext) -> list[SquareClassification]:
    return finalize_board(ctx.squares, ctx.occupancy, ctx.config.occupancy)[0]
