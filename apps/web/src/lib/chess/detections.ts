import { castlingRightsToFenField, inferCastlingRights } from "@/lib/chess/game";

export type BoardOrientation = "white" | "black";

export type BoardPiece =
  | "wP"
  | "wN"
  | "wB"
  | "wR"
  | "wQ"
  | "wK"
  | "bP"
  | "bN"
  | "bB"
  | "bR"
  | "bQ"
  | "bK";

export interface PieceDetection {
  square: string;
  piece: string;
  confidence: number;
  occupied?: boolean;
}

const LABEL_TO_FEN: Record<string, string> = {
  white_pawn: "P",
  white_knight: "N",
  white_bishop: "B",
  white_rook: "R",
  white_queen: "Q",
  white_king: "K",
  black_pawn: "p",
  black_knight: "n",
  black_bishop: "b",
  black_rook: "r",
  black_queen: "q",
  black_king: "k",
};

const LABEL_TO_BOARD: Record<string, BoardPiece> = {
  white_pawn: "wP",
  white_knight: "wN",
  white_bishop: "wB",
  white_rook: "wR",
  white_queen: "wQ",
  white_king: "wK",
  black_pawn: "bP",
  black_knight: "bN",
  black_bishop: "bB",
  black_rook: "bR",
  black_queen: "bQ",
  black_king: "bK",
};

const FILES = "abcdefgh";

export function isOccupiedDetection(det: PieceDetection): boolean {
  if (det.occupied !== undefined) return det.occupied;
  return det.piece !== "empty" && det.piece !== "unknown";
}

export function detectionsFromSquares(
  squares: {
    squareName?: string;
    name?: string;
    label: string;
    confidence: number;
    occupied?: boolean;
  }[],
): PieceDetection[] {
  return squares.map((sq) => ({
    square: sq.squareName ?? sq.name ?? "",
    piece: sq.occupied === false ? "empty" : sq.label,
    confidence: sq.confidence,
    occupied: sq.occupied,
  }));
}

export function buildPlacement(detections: PieceDetection[]): string {
  const bySquare = new Map(detections.map((d) => [d.square, d]));
  const ranks: string[] = [];

  for (let rank = 8; rank >= 1; rank -= 1) {
    let row = "";
    let emptyRun = 0;

    for (let fileIdx = 0; fileIdx < 8; fileIdx += 1) {
      const square = `${FILES[fileIdx]}${rank}`;
      const det = bySquare.get(square);
      const symbol =
        det && isOccupiedDetection(det) ? LABEL_TO_FEN[det.piece] : undefined;

      if (!symbol) {
        emptyRun += 1;
      } else {
        if (emptyRun > 0) {
          row += String(emptyRun);
          emptyRun = 0;
        }
        row += symbol;
      }
    }

    if (emptyRun > 0) row += String(emptyRun);
    ranks.push(row);
  }

  return ranks.join("/");
}

export function buildFen(
  detections: PieceDetection[],
  options?: { activeColor?: "w" | "b"; orientation?: BoardOrientation },
): string {
  const placement = buildPlacement(detections);
  const active = options?.activeColor ?? inferActiveColor(detections);
  const castling = castlingRightsToFenField(
    inferCastlingRights(`${placement} ${active} - - 0 1`),
  );
  return `${placement} ${active} ${castling} - 0 1`;
}

function inferActiveColor(detections: PieceDetection[]): "w" | "b" {
  const whitePieces = detections.filter(
    (d) => isOccupiedDetection(d) && d.piece.startsWith("white_"),
  ).length;
  const blackPieces = detections.filter(
    (d) => isOccupiedDetection(d) && d.piece.startsWith("black_"),
  ).length;
  return whitePieces >= blackPieces ? "w" : "b";
}

export function detectionsToBoardPosition(detections: PieceDetection[]): Record<string, BoardPiece> {
  const position: Record<string, BoardPiece> = {};

  for (const det of detections) {
    if (!isOccupiedDetection(det)) continue;
    const piece = LABEL_TO_BOARD[det.piece];
    if (piece) position[det.square] = piece;
  }

  return position;
}

export function lowConfidenceSquares(
  detections: PieceDetection[],
  threshold = 0.5,
): string[] {
  return detections
    .filter((d) => isOccupiedDetection(d) && d.confidence < threshold)
    .map((d) => d.square);
}
