import type { YoloBBox, YoloDetection } from "./types";

/** Port of apps/api/vision/scanner/square_assignment.py */

export interface AssignedYoloDetection extends YoloDetection {
  square: string;
}

export function bboxCenter(bbox: YoloBBox): [number, number] {
  return [bbox.x + bbox.w / 2, bbox.y + bbox.h / 2];
}

export function centerToSquareName(cx: number, cy: number, boardSize: number): string {
  const cell = boardSize / 8;
  const col = Math.min(7, Math.max(0, Math.floor(cx / cell)));
  const row = Math.min(7, Math.max(0, Math.floor(cy / cell)));
  const rank = 8 - row;
  const fileChar = String.fromCharCode("a".charCodeAt(0) + col);
  return `${fileChar}${rank}`;
}

export function assignSquares(
  detections: YoloDetection[],
  boardSize: number,
): AssignedYoloDetection[] {
  return detections.map((det) => {
    const [cx, cy] = bboxCenter(det.bbox);
    return { ...det, square: centerToSquareName(cx, cy, boardSize) };
  });
}

/** One piece per square — highest confidence wins (yolo_pieces._resolve_assignments). */
export function resolveSquareAssignments(
  detections: YoloDetection[],
  boardSize: number,
): Map<string, AssignedYoloDetection> {
  const withSquares = assignSquares(detections, boardSize);
  const best = new Map<string, AssignedYoloDetection>();
  for (const det of withSquares) {
    const prev = best.get(det.square);
    if (!prev || det.confidence > prev.confidence) {
      best.set(det.square, det);
    }
  }
  return best;
}
