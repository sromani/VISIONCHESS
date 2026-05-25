import { detectionsFromSquares } from "@/lib/chess/detections";
import type { OfflineGeometryResult } from "@/lib/vision/offlineBoardGeometry";
import {
  oneDetectionPerSquare,
  placementFenFromAssignments,
  validatePlacementFen,
} from "@/lib/vision/offlineYolo/fenFromYolo";
import type { NativeOfflineRecognizeResult } from "@/lib/vision/offlineYolo/types";
import type { DetectionResult, SquareInfo } from "@/types";

export function mapNativeYoloToDetection(
  geo: OfflineGeometryResult,
  native: NativeOfflineRecognizeResult,
  originalUrl: string,
): DetectionResult {
  const assigned = oneDetectionPerSquare(native.detections);
  const placement = native.placementFen || placementFenFromAssignments(assigned);
  const validation = validatePlacementFen(placement);

  const squares: SquareInfo[] = native.squares.length
    ? native.squares.map((sq) => ({
        name: sq.name,
        filename: `${sq.name}.jpg`,
        cellBbox: [0, 0, 0, 0] as SquareInfo["cellBbox"],
        cropBbox: [0, 0, 0, 0] as SquareInfo["cropBbox"],
        label: sq.label,
        confidence: sq.confidence,
        occupied: sq.occupied,
      }))
    : [...assigned.values()].map((sq) => ({
        name: sq.name,
        filename: `${sq.name}.jpg`,
        cellBbox: [0, 0, 0, 0] as SquareInfo["cellBbox"],
        cropBbox: sq.bbox as SquareInfo["cropBbox"],
        label: sq.label,
        confidence: sq.confidence,
        occupied: sq.occupied,
      }));

  const overlayUrl = native.debug.overlayJpegBase64.startsWith("data:")
    ? native.debug.overlayJpegBase64
    : `data:image/jpeg;base64,${native.debug.overlayJpegBase64}`;

  return {
    jobId: `offline-native-${Date.now()}`,
    confidence: validation.confidence,
    originalUrl,
    warpedUrl: geo.warpedUrl,
    overlayUrl: geo.debug.rectifiedGrid ?? geo.warpedUrl,
    corners: geo.corners,
    fen: validation.fen,
    interactiveFen: validation.interactiveFen,
    fenValid: validation.fenValid,
    boardReady: validation.boardReady,
    orientation: "standard",
    originalWidth: geo.originalWidth,
    originalHeight: geo.originalHeight,
    outputWidth: geo.outputWidth,
    outputHeight: geo.outputHeight,
    squares,
    processingMs: geo.processingMs + native.timings.totalMs,
    debug: {
      ...geo.debug,
      mlPieceTop1: overlayUrl,
    },
    metadata: {
      ...geo.metadata,
      geometryOnly: false,
      offline_native_yolo: true,
      piece_backend: "yolo_object_detection",
      geometry_backend: geo.metadata.geometry_backend,
      timings: {
        geometryMs: geo.processingMs,
        ...native.timings,
      },
      offline_log: native.debug.logLines,
      yolo_detections: native.detections,
      fen_validation: {
        is_valid: validation.fenValid,
        kings_ok: validation.kingsOk,
        piece_count: validation.pieceCount,
      },
      board_ready: validation.boardReady,
      fen_valid: validation.fenValid,
    },
  };
}

export function squaresToDetectionsForUi(detection: DetectionResult) {
  return detectionsFromSquares(
    detection.squares.map((sq) => ({
      name: sq.name,
      label: sq.label ?? "empty",
      confidence: sq.confidence ?? 0,
      occupied: sq.occupied,
    })),
  );
}
