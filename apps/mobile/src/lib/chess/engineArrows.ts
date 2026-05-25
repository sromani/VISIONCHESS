import type { Square } from "react-chessboard/dist/chessboard/types";
import type { AnalysisResult, EngineLine } from "@/types";

/** Lichess-style engine green — single hue, opacity by rank */
export const ENGINE_ARROW_RGB = "80, 220, 120";

export const ENGINE_ARROW_OPACITIES = [1, 0.65, 0.35] as const;

export function engineArrowColor(index: number): string {
  const opacity =
    ENGINE_ARROW_OPACITIES[Math.min(index, ENGINE_ARROW_OPACITIES.length - 1)];
  return `rgba(${ENGINE_ARROW_RGB}, ${opacity})`;
}

/** Precomputed for legend dots and arrow props */
export const ENGINE_ARROW_COLORS = [
  engineArrowColor(0),
  engineArrowColor(1),
  engineArrowColor(2),
] as const;

export const ENGINE_SQUARE_BEST_GLOW = `rgba(${ENGINE_ARROW_RGB}, 0.14)`;
export const ENGINE_SQUARE_BEST_GLOW_SHADOW = `inset 0 0 14px rgba(${ENGINE_ARROW_RGB}, 0.22)`;
export const ENGINE_SQUARE_HOVER_FROM = `rgba(${ENGINE_ARROW_RGB}, 0.32)`;
export const ENGINE_SQUARE_HOVER_TO = `rgba(${ENGINE_ARROW_RGB}, 0.2)`;

/** react-chessboard Arrow: [from, to, color?] */
export type BoardArrow = [Square, Square, string?];

export function uciToSquares(uci: string): { from: string; to: string } {
  return { from: uci.slice(0, 2), to: uci.slice(2, 4) };
}

export function buildEngineArrows(
  lines: EngineLine[],
  maxCount: number,
): BoardArrow[] {
  return lines
    .slice(0, maxCount)
    .filter((line) => line.from && line.to && line.from !== line.to)
    .map((line, index) => [
      line.from as Square,
      line.to as Square,
      engineArrowColor(index),
    ]);
}

export function getEngineLines(analysis: AnalysisResult | null, maxCount: number): EngineLine[] {
  if (!analysis?.lines.length) return [];
  return analysis.lines.slice(0, maxCount);
}
