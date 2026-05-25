"""Stockfish analysis service."""

from __future__ import annotations

import asyncio
import uuid

import chess
import chess.engine

from app.core.exceptions import EngineUnavailableError
from app.core.logging import get_logger
from app.core.settings import Settings
from app.models.analysis import AnalysisResult, EngineLine

logger = get_logger(__name__)


class AnalysisService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def analyze(self, fen: str, depth: int | None = None, multipv: int | None = None) -> AnalysisResult:
        depth = depth or self._settings.stockfish_depth_default
        multipv = multipv or self._settings.stockfish_multipv_default

        try:
            chess.Board(fen)
        except ValueError as exc:
            raise EngineUnavailableError(f"Invalid FEN: {exc}") from exc

        logger.info("analysis_started", fen=fen, depth=depth, multipv=multipv)
        try:
            result = await asyncio.to_thread(self._run_engine, fen, depth, multipv)
        except Exception as exc:  # noqa: BLE001
            logger.exception("analysis_failed", error=str(exc))
            raise EngineUnavailableError(str(exc)) from exc

        logger.info("analysis_completed", best_move=result.best_move, processing_ms=result.processing_ms)
        return result

    def _run_engine(self, fen: str, depth: int, multipv: int) -> AnalysisResult:
        import time

        started = time.perf_counter()
        with chess.engine.SimpleEngine.popen_uci(self._settings.stockfish_path) as engine:
            board = chess.Board(fen)
            limit = chess.engine.Limit(depth=depth)
            infos = engine.analyse(board, limit, multipv=multipv)

            lines: list[EngineLine] = []
            for info in infos:
                pv_moves = info.get("pv", [])
                if not pv_moves:
                    continue
                score = info.get("score")
                if score is None:
                    continue

                white = score.white()
                cp = None if white.is_mate() else white.score()
                mate = white.mate() if white.is_mate() else None
                lines.append(
                    EngineLine(
                        move=pv_moves[0].uci(),
                        eval_cp=cp,
                        eval_mate=mate,
                        pv=[m.uci() for m in pv_moves],
                    )
                )

            elapsed = int((time.perf_counter() - started) * 1000)
            primary = lines[0] if lines else EngineLine(move="", eval_cp=None, eval_mate=None, pv=[])

            return AnalysisResult(
                id=str(uuid.uuid4()),
                fen=fen,
                depth=depth,
                best_move=primary.move,
                evaluation_cp=primary.eval_cp,
                evaluation_mate=primary.eval_mate,
                lines=lines,
                processing_ms=elapsed,
            )
