"""Phases 6–7 — hypothesis ranking, python-chess validation, Stockfish scoring."""

from __future__ import annotations

import cv2
import numpy as np

from vision.classification.pipeline import grid_to_matrix, _is_board_ready, _squares_to_grid
from vision.classification.types import ClassificationResult, OrientationCandidate
from vision.hypotheses.engine import HypothesisEngine
from vision.occupancy.finalize import finalize_board
from vision.scanner.context import ScanContext


def run_validation(ctx: ScanContext) -> ClassificationResult:
    if not ctx.squares:
        msg = "Classification must run before validation"
        raise ValueError(msg)

    finalized, board_threshold, prior_applied = finalize_board(
        ctx.squares,
        ctx.occupancy,
        ctx.config.occupancy,
    )
    ctx.squares = finalized
    ctx.metadata["occupancy_finalize"] = {
        "board_threshold": board_threshold,
        "board_prior_applied": prior_applied,
        "occupied_count": sum(1 for sq in finalized if sq.occupied),
    }

    cfg = ctx.config.classification
    ml_cfg = ctx.config.ml
    stockfish_path = cfg.stockfish_path if ctx.config.use_stockfish_scoring else None

    engine = HypothesisEngine(
        stockfish_path=stockfish_path,
        use_stockfish=ml_cfg.use_stockfish_scoring and bool(stockfish_path),
        allow_partial_fen=cfg.allow_partial_fen,
    )
    hypotheses = engine.generate(finalized)
    ctx.metadata["hypotheses"] = [
        {
            "name": h.name,
            "fen": h.fen,
            "legality_score": h.legality_score,
            "stockfish_bonus": h.stockfish_bonus,
            "total_score": h.total_score,
            "is_valid": h.is_valid,
        }
        for h in hypotheses
    ]

    best_hyp = hypotheses[0] if hypotheses else None
    if best_hyp is None:
        msg = "No board hypotheses generated"
        raise ValueError(msg)

    pred_grid = _squares_to_grid(finalized)
    candidates: list[OrientationCandidate] = []
    for h in hypotheses:
        if h.name == "flipped":
            from vision.classification.orientation import flip_grid_vertical

            grid = flip_grid_vertical(pred_grid)
        else:
            grid = pred_grid
        candidates.append(
            OrientationCandidate(
                name=h.name,
                grid=grid,
                fen=h.fen,
                legality_score=h.total_score,
                fen_confidence=h.fen_confidence,
                is_valid=h.is_valid,
                active_color=h.active_color,
            )
        )
    ctx.candidates = candidates
    best = candidates[0]

    from vision.classification.legality import score_grid

    legality = score_grid(
        best.grid,
        active_color=best.active_color,
        allow_partial=cfg.allow_partial_fen,
    )
    board_ready = _is_board_ready(legality, cfg)

    result = ClassificationResult(
        fen=legality.fen_result.fen,
        placement=legality.fen_result.placement,
        confidence=legality.fen_result.confidence,
        is_valid=legality.is_valid,
        board_ready=board_ready,
        interactive_fen=legality.fen_result.fen if board_ready else None,
        board_matrix=grid_to_matrix(legality.fen_result.repaired_grid),
        orientation=best.name,
        active_color=best.active_color,
        squares=tuple(finalized),
        grid=legality.fen_result.repaired_grid,
        candidates=tuple(candidates),
        classifier_backend=ctx.metadata.get("classification", {}).get("backend", "heuristic"),
        dataset_grid=ctx.dataset_grid,
    )

    ctx.metadata["validation"] = {
        "orientation": best.name,
        "fen_valid": legality.is_valid,
        "board_ready": board_ready,
        "candidates": [c.to_metadata() for c in candidates],
        "hypothesis_engine": "chesscog_style",
    }

    if ctx.config.collect_debug:
        ctx.add_debug("fen_candidates", _render_hypotheses_panel(hypotheses))
        ctx.add_debug("classifier_confidence", _render_confidence_grid(ctx))

    return result


def _render_hypotheses_panel(hypotheses, width: int = 720, height: int = 200) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)
    cv2.putText(panel, "BOARD HYPOTHESES (legality + Stockfish)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
    y = 58
    for h in sorted(hypotheses, key=lambda x: x.total_score, reverse=True):
        line = (
            f"{h.name}: total={h.total_score:.1f} "
            f"(leg={h.legality_score:.1f}+sf={h.stockfish_bonus:.1f}) "
            f"valid={h.is_valid} {h.fen[:40]}"
        )
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 180), 1, cv2.LINE_AA)
        y += 22
    return panel


def _render_confidence_grid(ctx: ScanContext, cell_px: int = 64) -> np.ndarray:
    board = np.zeros((cell_px * 8, cell_px * 8, 3), dtype=np.uint8)
    for sq in ctx.squares:
        conf = sq.confidence if sq.occupied else sq.occupancy_score
        intensity = int(min(255, max(40, conf * 255)))
        color = (intensity // 3, intensity // 2, intensity)
        y0, x0 = sq.row * cell_px, sq.col * cell_px
        board[y0 : y0 + cell_px, x0 : x0 + cell_px] = color
        label = sq.label[:3] if sq.occupied else "—"
        cv2.putText(
            board,
            f"{label} {conf:.2f}",
            (x0 + 2, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return board
