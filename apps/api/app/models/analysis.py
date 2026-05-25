"""Analysis domain models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineLine:
    move: str
    eval_cp: int | None
    eval_mate: int | None
    pv: list[str]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    id: str
    fen: str
    depth: int
    best_move: str
    evaluation_cp: int | None
    evaluation_mate: int | None
    lines: list[EngineLine]
    processing_ms: int
