import { orderPointsYSort } from "../geometry/orderPoints";
import type { Point2 } from "../geometry/types";
import type { YoloBBox, YoloDetection } from "./types";

/** Min/max fraction of image area for a valid board bounding box. */
const MIN_BOARD_AREA_RATIO = 0.04;
const MAX_BOARD_AREA_RATIO = 0.92;
const BOARD_CLASS = "board";

export interface BoardRegionResult {
  corners: Point2[];
  bbox: YoloBBox;
  confidence: number;
  areaRatio: number;
}

/** Pick best YOLO "board" detection and convert axis-aligned bbox → quad (TL, TR, BR, BL). */
export function boardRegionFromDetections(
  detections: YoloDetection[],
  imageWidth: number,
  imageHeight: number,
  options?: { padding?: number; minConfidence?: number },
): BoardRegionResult | null {
  const padding = options?.padding ?? 0.03;
  const minConfidence = options?.minConfidence ?? 0.2;
  const imgArea = imageWidth * imageHeight;

  const boardDets = detections.filter(
    (d) => d.className === BOARD_CLASS && d.confidence >= minConfidence,
  );
  if (boardDets.length === 0) return null;

  let best: YoloDetection | null = null;
  let bestScore = -1;

  for (const det of boardDets) {
    const area = det.bbox.w * det.bbox.h;
    const ratio = area / imgArea;
    if (ratio < MIN_BOARD_AREA_RATIO || ratio > MAX_BOARD_AREA_RATIO) continue;
    const score = area * det.confidence;
    if (score > bestScore) {
      bestScore = score;
      best = det;
    }
  }

  if (!best) return null;

  const corners = bboxToCorners(best.bbox, imageWidth, imageHeight, padding);
  return {
    corners,
    bbox: best.bbox,
    confidence: best.confidence,
    areaRatio: (best.bbox.w * best.bbox.h) / imgArea,
  };
}

export function bboxToCorners(
  bbox: YoloBBox,
  imageWidth: number,
  imageHeight: number,
  padding = 0.03,
): Point2[] {
  const padX = bbox.w * padding;
  const padY = bbox.h * padding;
  const x1 = Math.max(0, Math.round(bbox.x - padX));
  const y1 = Math.max(0, Math.round(bbox.y - padY));
  const x2 = Math.min(imageWidth, Math.round(bbox.x + bbox.w + padX));
  const y2 = Math.min(imageHeight, Math.round(bbox.y + bbox.h + padY));
  return orderPointsYSort([
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x2, y: y2 },
    { x: x1, y: y2 },
  ]);
}
