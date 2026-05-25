import { API_BASE, LOCAL_GEOMETRY_FIRST, OFFLINE_MODE, USE_REMOTE_VISION, VISION_LOCAL } from "@/lib/config";
import { detectBoard } from "@/lib/api/detectBoard";
import { ApiError } from "@/lib/api/client";
import { detectBoardLocal } from "@/vision/detectBoardLocal";
import { recognizePiecesNative } from "@/lib/vision/offlineYolo/nativeBridge";
import { rectifyBoardFromFile } from "@/lib/vision/offlineBoardGeometry";
import type { DetectionResult } from "@/types";

export const START_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export type ProgressCallback = (
  step: "upload" | "detect" | "classify" | "validate" | "analyze",
  detail?: string,
) => void;

export type VisionPipelineResult =
  | { mode: "remote"; detection: DetectionResult }
  | { mode: "local"; detection: DetectionResult }
  | { mode: "offline_stub"; detection: DetectionResult };

export async function runVisionPipeline(
  file: File,
  onProgress?: ProgressCallback,
): Promise<VisionPipelineResult> {
  const originalUrl = URL.createObjectURL(file);
  onProgress?.("upload");

  if (VISION_LOCAL) {
    onProgress?.("detect");
    const detection = await detectBoardLocal(file);
    onProgress?.("classify");
    onProgress?.("validate");
    onProgress?.("analyze");
    return { mode: "local", detection };
  }

  if (USE_REMOTE_VISION) {
    if (!LOCAL_GEOMETRY_FIRST) {
      const { detection } = await runRemotePipeline(file, originalUrl, onProgress);
      return { mode: "remote", detection };
    }
    const { detection } = await runLocalGeometryPipeline(file, originalUrl, onProgress);
    return { mode: "remote", detection };
  }

  const { detection } = await runLocalGeometryPipeline(file, originalUrl, onProgress);
  return { mode: "offline_stub", detection };
}

async function runRemotePipeline(
  file: File,
  originalUrl: string,
  onProgress?: ProgressCallback,
): Promise<{ detection: DetectionResult }> {
  onProgress?.("detect");

  let detection: DetectionResult;
  try {
    detection = await detectBoard(file, originalUrl, "/detect-lc2fen");
  } catch (err) {
    if (err instanceof ApiError) {
      throw new Error(err.message);
    }
    if (err instanceof TypeError) {
      throw new Error("Could not reach detection server.");
    }
    throw err;
  }

  onProgress?.("classify");
  onProgress?.("validate");
  onProgress?.("analyze");

  return { detection };
}

/** M1: canvas_fast warp → native iOS YOLO (offline) or hybrid API for pieces. */
async function runLocalGeometryPipeline(
  file: File,
  originalUrl: string,
  onProgress?: ProgressCallback,
): Promise<{ detection: DetectionResult }> {
  onProgress?.("detect", "Detecting board edges…");

  const geo = await rectifyBoardFromFile(file);

  const nativeDetection = await recognizePiecesNative(geo, originalUrl);
  if (nativeDetection) {
    onProgress?.("classify");
    onProgress?.("validate");
    onProgress?.("analyze");
    return { detection: nativeDetection };
  }

  let detection: DetectionResult = {
    jobId: `offline-${Date.now()}`,
    confidence: geo.confidence,
    originalUrl,
    warpedUrl: geo.warpedUrl,
    overlayUrl: geo.debug.rectifiedGrid ?? geo.warpedUrl,
    corners: geo.corners,
    fen: "",
    interactiveFen: null,
    fenValid: false,
    boardReady: false,
    orientation: "standard",
    originalWidth: geo.originalWidth,
    originalHeight: geo.originalHeight,
    outputWidth: geo.outputWidth,
    outputHeight: geo.outputHeight,
    squares: [],
    processingMs: geo.processingMs,
    debug: geo.debug,
    metadata: geo.metadata,
  };

  if (API_BASE && !OFFLINE_MODE) {
    onProgress?.("classify", "Detecting pieces (YOLO)…");
    try {
      const apiDetection = await detectBoard(file, originalUrl, "/detect-lc2fen");
      detection = mergeLocalGeometryWithApi(geo, apiDetection, originalUrl);
      onProgress?.("validate");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Piece detection failed";
      detection = {
        ...detection,
        metadata: {
          ...detection.metadata,
          piece_detection_error: message,
        },
      };
    }
  }

  onProgress?.("analyze");
  return { detection };
}

function mergeLocalGeometryWithApi(
  geo: Awaited<ReturnType<typeof rectifyBoardFromFile>>,
  api: DetectionResult,
  originalUrl: string,
): DetectionResult {
  return {
    ...api,
    originalUrl,
    warpedUrl: geo.warpedUrl,
    overlayUrl: geo.debug.rectifiedGrid ?? api.overlayUrl,
    corners: geo.corners.length >= 4 ? geo.corners : api.corners,
    debug: {
      ...api.debug,
      ...geo.debug,
      rectifiedBoard: geo.debug.rectifiedBoard ?? api.debug?.rectifiedBoard,
      rectifiedGrid: geo.debug.rectifiedGrid ?? api.debug?.rectifiedGrid,
    },
    processingMs: geo.processingMs + api.processingMs,
    metadata: {
      ...api.metadata,
      geometryOnly: false,
      geometry_backend: geo.metadata?.geometry_backend ?? "canvas_fast",
      local_geometry_ms: geo.processingMs,
      api_geometry_backend: api.metadata?.geometry_backend,
    },
  };
}
