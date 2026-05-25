/** Port of yolo_pieces._fen_from_assignments + fen_validate (placement only). */

import {
  castlingRightsToFenField,
  inferCastlingRights,
  tryCreateGame,
} from "@/lib/chess/game";
import type { BoardMatrixCell, SquareInfo } from "@/types";

import type { AssignedYoloDetection } from "./squareAssignment";

const YOLO_TO_FEN: Record<string, string> = {
  white_king: "K",
  white_queen: "Q",
  white_rook: "R",
  white_bishop: "B",
  white_knight: "N",
  white_pawn: "P",
  black_king: "k",
  black_queen: "q",
  black_rook: "r",
  black_bishop: "b",
  black_knight: "n",
  black_pawn: "p",
};

const FILES = "abcdefgh";

export interface FenValidationResult {
  placement: string;
  interactiveFen: string | null;
  isValid: boolean;
  boardReady: boolean;
  confidence: number;
  kingsOk: boolean;
  pieceCount: number;
}

export function placementFromAssignments(assigned: Map<string, AssignedYoloDetection>): string {
  const ranks: string[] = [];
  for (let rank = 8; rank >= 1; rank -= 1) {
    let row = "";
    let emptyRun = 0;
    for (let fileIdx = 0; fileIdx < 8; fileIdx += 1) {
      const name = `${FILES[fileIdx]}${rank}`;
      const det = assigned.get(name);
      const symbol = det ? YOLO_TO_FEN[det.className] : undefined;
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

export function validatePlacementFen(
  placement: string,
  options?: { minPieces?: number; minConfidence?: number },
): FenValidationResult {
  const minPieces = options?.minPieces ?? 4;
  const minConfidence = options?.minConfidence ?? 0.35;
  const pieceCount = [...placement].filter((ch) => /[a-zA-Z]/.test(ch)).length;
  const confidence = Math.min(1, pieceCount / 32);

  let kingsOk = false;
  let isValid = false;
  let interactiveFen: string | null = null;

  const draft = tryCreateGame(`${placement} w - - 0 1`);
  if (draft) {
    let wKings = 0;
    let bKings = 0;
    for (const row of draft.board()) {
      for (const cell of row) {
        if (cell?.type !== "k") continue;
        if (cell.color === "w") wKings += 1;
        else bKings += 1;
      }
    }
    kingsOk = wKings === 1 && bKings === 1;
    const rights = castlingRightsToFenField(inferCastlingRights(`${placement} w - - 0 1`));
    const full = `${placement} w ${rights} - 0 1`;
    if (tryCreateGame(full)) {
      isValid = true;
      interactiveFen = full;
    }
  }

  const boardReady =
    isValid && kingsOk && pieceCount >= minPieces && confidence >= minConfidence;
  if (!boardReady) {
    interactiveFen = null;
  }

  return {
    placement,
    interactiveFen: boardReady ? interactiveFen : null,
    isValid,
    boardReady,
    confidence,
    kingsOk,
    pieceCount,
  };
}

export function buildBoardMatrix(
  assigned: Map<string, AssignedYoloDetection>,
): BoardMatrixCell[][] {
  const matrix: BoardMatrixCell[][] = [];
  for (let row = 0; row < 8; row += 1) {
    const rank = 8 - row;
    const line: BoardMatrixCell[] = [];
    for (let col = 0; col < 8; col += 1) {
      const name = `${FILES[col]}${rank}`;
      const det = assigned.get(name);
      line.push({
        label: det?.className ?? "empty",
        confidence: det?.confidence ?? 0,
      });
    }
    matrix.push(line);
  }
  return matrix;
}

export function buildSquareInfos(
  boardSize: number,
  assigned: Map<string, AssignedYoloDetection>,
): SquareInfo[] {
  const cell = boardSize / 8;
  const squares: SquareInfo[] = [];
  for (let row = 0; row < 8; row += 1) {
    const rank = 8 - row;
    for (let col = 0; col < 8; col += 1) {
      const name = `${FILES[col]}${rank}`;
      const det = assigned.get(name);
      const x = Math.round(col * cell);
      const y = Math.round(row * cell);
      const w = Math.round(cell);
      const h = Math.round(cell);
      squares.push({
        name,
        filename: `${name}.jpg`,
        cellBbox: [x, y, w, h],
        cropBbox: [x, y, w, h],
        label: det?.className ?? "empty",
        confidence: det?.confidence ?? 0,
        occupied: det != null,
      });
    }
  }
  return squares;
}
