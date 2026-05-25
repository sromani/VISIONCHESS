import type { DetectionResult } from "@/types";

import { drawCornerOverlay, imageDataToObjectUrl } from "./geometryClient";
import type { RectifyResult } from "./geometry/types";
import {
  buildBoardMatrix,
  buildSquareInfos,
  placementFromAssignments,
  validatePlacementFen,
} from "./yolo/fenFromYolo";
import { bboxCenter, resolveSquareAssignments } from "./yolo/squareAssignment";
import type { ImageRGBA, LocalYoloResult } from "./yolo/types";

export interface LocalDetectContext {
  originalUrl: string;
  originalWidth: number;
  originalHeight: number;
  originalImage: ImageRGBA;
  geometry: RectifyResult;
  /** Optional pre-rendered overlay (e.g. localization debug). */
  overlayUrl?: string;
}

/**
 * Map local YOLO on rectified board → DetectionResult (same contract as /detect-lc2fen).
 */
export function localYoloToDetection(
  local: LocalYoloResult,
  ctx: LocalDetectContext,
): DetectionResult {
  const boardSize = Math.min(local.imageWidth, local.imageHeight);
  const assigned = resolveSquareAssignments(local.detections, boardSize);
  const placement = placementFromAssignments(assigned);
  const validation = validatePlacementFen(placement);

  const yoloRecords = [...assigned.values()].map((det) => ({
    label: det.className,
    confidence: Math.round(det.confidence * 10000) / 10000,
    square: det.square,
    bbox: [det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h],
    center: bboxCenter(det.bbox).map((v) => Math.round(v * 10) / 10),
  }));

  const warpedUrl = imageDataToObjectUrl(ctx.geometry.warpedPreview);
  const overlayUrl =
    ctx.overlayUrl ??
    (ctx.geometry.corners.length === 4
      ? drawCornerOverlay(ctx.originalImage, ctx.geometry.corners)
      : ctx.originalUrl);

  const totalMs =
    Math.round(ctx.geometry.metrics.geometryMs) + Math.round(local.metrics.totalMs);

  return {
    jobId: `local-yolo-${Date.now()}`,
    confidence: validation.confidence,
    originalUrl: ctx.originalUrl,
    warpedUrl,
    overlayUrl,
    corners: ctx.geometry.corners.map((c) => ({ x: c.x, y: c.y })),
    fen: placement,
    interactiveFen: validation.interactiveFen,
    fenConfidence: validation.confidence,
    fenValid: validation.isValid,
    boardReady: validation.boardReady,
    boardMatrix: buildBoardMatrix(assigned),
    orientation: "white",
    originalWidth: ctx.originalWidth,
    originalHeight: ctx.originalHeight,
    outputWidth: local.imageWidth,
    outputHeight: local.imageHeight,
    squares: buildSquareInfos(boardSize, assigned),
    processingMs: totalMs,
    debug: null,
    metadata: {
      offline: true,
      visionLocal: true,
      backend: "local_lc2fen_geometry_yolo",
      geometryOk: ctx.geometry.geometryOk,
      board_found: ctx.geometry.boardFound,
      localization_status: ctx.geometry.localizationStatus,
      geometryMethod: ctx.geometry.metrics.method,
      geometryMs: ctx.geometry.metrics.geometryMs,
      lapsHits: ctx.geometry.metrics.lapsHits,
      boardConfidence: ctx.geometry.metrics.boardConfidence,
      pieceCount: validation.pieceCount,
      kingsOk: validation.kingsOk,
      yolo_detections: yoloRecords,
      yolo_metrics: local.metrics,
      modelPath: local.modelPath,
    },
  };
}
