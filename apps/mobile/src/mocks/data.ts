import { AnalysisResult, DetectionResult } from "@/types";

export const MOCK_FEN =
  "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4";

export const MOCK_DETECTION_META: Omit<
  DetectionResult,
  "originalUrl" | "warpedUrl" | "overlayUrl" | "jobId"
> = {
  confidence: 0.91,
  corners: [
    { x: 82, y: 44 },
    { x: 518, y: 28 },
    { x: 492, y: 556 },
    { x: 38, y: 524 },
  ],
  fen: MOCK_FEN,
  interactiveFen: MOCK_FEN,
  boardReady: true,
  fenValid: true,
  originalWidth: 1200,
  originalHeight: 900,
  outputWidth: 800,
  outputHeight: 800,
  processingMs: 840,
  squares: buildMockSquares(),
};

export const MOCK_ANALYSIS: AnalysisResult = {
  evaluationCp: 156,
  evaluationMate: null,
  bestMove: "h5f7",
  depth: 18,
  processingMs: 620,
  lines: [
    {
      multipv: 1,
      move: "h5f7",
      from: "h5",
      to: "f7",
      san: "Qxf7+",
      evalCp: 156,
      evalMate: null,
      pv: ["h5f7", "e8f7", "c4d5"],
    },
    {
      multipv: 2,
      move: "c4f7",
      from: "c4",
      to: "f7",
      san: "Bxf7+",
      evalCp: 98,
      evalMate: null,
      pv: ["c4f7", "e8f7", "d1f3"],
    },
    {
      multipv: 3,
      move: "h5f5",
      from: "h5",
      to: "f5",
      san: "Qf5",
      evalCp: 72,
      evalMate: null,
      pv: ["h5f5", "d8f6", "d1f3"],
    },
  ],
};

function buildMockSquares() {
  const squares = [];
  for (let rank = 8; rank >= 1; rank--) {
    for (const file of "abcdefgh") {
      const col = file.charCodeAt(0) - 97;
      const row = 8 - rank;
      const cell = col * 100;
      const cellY = row * 100;
      squares.push({
        name: `${file}${rank}`,
        filename: `${file}${rank}.png`,
        cellBbox: [cell, cellY, cell + 100, cellY + 100] as [number, number, number, number],
        cropBbox: [cell + 8, cellY + 8, cell + 92, cellY + 92] as [number, number, number, number],
      });
    }
  }
  return squares;
}
