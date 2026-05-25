import { Chess } from "chess.js";

import { castlingRightsToFenField, inferCastlingRights } from "@/lib/chess/game";
import { YOLO_LABEL_TO_FEN } from "@/lib/vision/offlineYolo/yoloConfig";

export interface YoloSquareAssignment {
  name: string;
  label: string;
  confidence: number;
  occupied: boolean;
  bbox: [number, number, number, number];
}

/** Port of `yolo_pieces._fen_from_assignments`. */
export function placementFenFromAssignments(
  assigned: Map<string, Pick<YoloSquareAssignment, "label">>,
): string {
  const ranks: string[] = [];
  for (let rank = 8; rank >= 1; rank -= 1) {
    let row = "";
    let emptyRun = 0;
    for (let fileIdx = 0; fileIdx < 8; fileIdx += 1) {
      const name = `${String.fromCharCode(97 + fileIdx)}${rank}`;
      const det = assigned.get(name);
      const symbol = det ? YOLO_LABEL_TO_FEN[det.label] : undefined;
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

/** Port of `fen_validate.validate_placement_fen` (chess.js instead of python-chess). */
export function validatePlacementFen(placement: string): {
  fen: string;
  interactiveFen: string | null;
  fenValid: boolean;
  boardReady: boolean;
  confidence: number;
  kingsOk: boolean;
  pieceCount: number;
} {
  const pieceCount = [...placement.replace(/[^a-zA-Z]/g, "")].length;
  const confidence = Math.min(1, pieceCount / 32);

  try {
    const draft = new Chess(`${placement} w - - 0 1`);
    const rights = castlingRightsToFenField(inferCastlingRights(draft.fen()));
    const interactiveFen = `${placement} w ${rights} - 0 1`;
    const board = new Chess(interactiveFen);

    let kingsOk = false;
    try {
      const b = board.board();
      let kings = 0;
      for (const row of b) {
        for (const p of row) {
          if (p?.type === "k") kings += 1;
        }
      }
      kingsOk = kings === 2;
    } catch {
      kingsOk = false;
    }

    let fenValid = true;
    try {
      board.move("a2a3");
      board.undo();
    } catch {
      fenValid = !board.isGameOver() || board.isCheckmate() || board.isStalemate();
    }

    const boardReady =
      fenValid &&
      kingsOk &&
      pieceCount >= 4 &&
      confidence >= 0.5;

    return {
      fen: placement,
      interactiveFen: boardReady ? board.fen() : null,
      fenValid,
      boardReady,
      confidence,
      kingsOk,
      pieceCount,
    };
  } catch {
    return {
      fen: placement,
      interactiveFen: null,
      fenValid: false,
      boardReady: false,
      confidence,
      kingsOk: false,
      pieceCount,
    };
  }
}

export function oneDetectionPerSquare(
  detections: Array<{
    label: string;
    confidence: number;
    bbox: [number, number, number, number];
    square: string;
  }>,
): Map<string, YoloSquareAssignment> {
  const best = new Map<string, YoloSquareAssignment>();
  for (const det of detections) {
    const prev = best.get(det.square);
    if (!prev || det.confidence > prev.confidence) {
      best.set(det.square, {
        name: det.square,
        label: det.label,
        confidence: det.confidence,
        occupied: det.label !== "empty" && det.label !== "piece",
        bbox: det.bbox,
      });
    }
  }
  return best;
}
