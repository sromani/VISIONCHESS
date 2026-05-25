import { Chess } from "chess.js";

import { uciToSquares } from "@/lib/chess/engineArrows";
import type { AnalysisResult, EngineLine } from "@/types";

export interface ParsedUciInfo {
  multipv: number;
  depth: number;
  evaluationCp: number | null;
  evaluationMate: number | null;
  nodes: number | null;
  nps: number | null;
  pv: string[];
  bestMoveUci: string;
}

export function parseUciInfoLine(line: string): ParsedUciInfo | null {
  if (!line.startsWith("info ") || line.includes(" currmove ")) return null;

  const depthMatch = line.match(/\bdepth (\d+)\b/);
  if (!depthMatch) return null;

  const depth = Number(depthMatch[1]);
  const multipvMatch = line.match(/\bmultipv (\d+)\b/);
  const multipv = multipvMatch ? Number(multipvMatch[1]) : 1;

  let evaluationCp: number | null = null;
  let evaluationMate: number | null = null;

  const cpMatch = line.match(/\bscore cp (-?\d+)\b/);
  const mateMatch = line.match(/\bscore mate (-?\d+)\b/);
  if (mateMatch) {
    evaluationMate = Number(mateMatch[1]);
  } else if (cpMatch) {
    evaluationCp = Number(cpMatch[1]);
  }

  const nodesMatch = line.match(/\bnodes (\d+)\b/);
  const npsMatch = line.match(/\bnps (\d+)\b/);
  const pvMatch = line.match(/\bpv (.+)$/);
  const pv = pvMatch ? pvMatch[1].trim().split(/\s+/) : [];

  return {
    multipv,
    depth,
    evaluationCp,
    evaluationMate,
    nodes: nodesMatch ? Number(nodesMatch[1]) : null,
    nps: npsMatch ? Number(npsMatch[1]) : null,
    pv,
    bestMoveUci: pv[0] ?? "",
  };
}

export function parseBestMoveLine(line: string): string | null {
  if (!line.startsWith("bestmove ")) return null;
  return line.split(/\s+/)[1] ?? null;
}

export function uciToSan(fen: string, uci: string): string {
  if (!uci || uci === "(none)") return "—";
  try {
    const game = new Chess(fen);
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const promotion = uci.length > 4 ? uci[4] : undefined;
    const move = game.move({ from, to, promotion });
    return move?.san ?? uci;
  } catch {
    return uci;
  }
}

/** UCI scores are from side-to-move; normalize to white's perspective for the UI. */
export function normalizeEvalToWhite(
  fen: string,
  cp: number | null,
  mate: number | null,
): { cp: number | null; mate: number | null } {
  const turn = fen.split(" ")[1];
  if (turn !== "b") return { cp, mate };
  return {
    cp: cp !== null ? -cp : null,
    mate: mate !== null ? -mate : null,
  };
}

function parsedInfoToEngineLine(fen: string, info: ParsedUciInfo): EngineLine {
  const { cp, mate } = normalizeEvalToWhite(
    fen,
    info.evaluationCp,
    info.evaluationMate,
  );
  const { from, to } = uciToSquares(info.bestMoveUci);

  return {
    multipv: info.multipv,
    move: info.bestMoveUci,
    from,
    to,
    san: info.bestMoveUci ? uciToSan(fen, info.bestMoveUci) : undefined,
    evalCp: cp,
    evalMate: mate,
    pv: info.pv,
  };
}

export function mergeMultiPvToAnalysisResult(
  fen: string,
  linesByPv: Map<number, ParsedUciInfo>,
  startedAt: number,
): AnalysisResult {
  const sorted = [...linesByPv.entries()].sort((a, b) => a[0] - b[0]);
  const engineLines = sorted
    .map(([, info]) => parsedInfoToEngineLine(fen, info))
    .filter((line) => line.move);

  const primaryInfo = sorted[0]?.[1];
  const primary = engineLines[0];
  const depth = Math.max(...sorted.map(([, info]) => info.depth), 0);
  const nodesEntry = sorted.find(([, info]) => info.nodes !== null);

  return {
    evaluationCp: primary?.evalCp ?? null,
    evaluationMate: primary?.evalMate ?? null,
    bestMove: primary?.move ?? "",
    bestMoveSan: primary?.san,
    depth: primaryInfo?.depth ?? depth,
    nodesSearched: nodesEntry?.[1].nodes ?? undefined,
    nps: primaryInfo?.nps ?? undefined,
    lines: engineLines,
    processingMs: Date.now() - startedAt,
  };
}

/** @deprecated use mergeMultiPvToAnalysisResult */
export function toAnalysisResult(
  info: ParsedUciInfo,
  fen: string,
  startedAt: number,
): AnalysisResult {
  return mergeMultiPvToAnalysisResult(fen, new Map([[info.multipv, info]]), startedAt);
}
