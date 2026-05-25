import { USE_REMOTE_VISION, VISION_LOCAL } from "@/lib/config";
import { detectBoard } from "@/lib/api/detectBoard";
import { ApiError } from "@/lib/api/client";
import { detectBoardLocal } from "@/vision/detectBoardLocal";
import { createImagePreview } from "@/lib/storage/imagePreview";
import type { DetectionResult } from "@/types";

export type ProgressCallback = (
  step: "upload" | "detect" | "classify" | "validate" | "analyze",
) => void;

const START_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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
    const { detection } = await runRemotePipeline(file, originalUrl, onProgress);
    return { mode: "remote", detection };
  }

  const { detection } = await runOfflineStubPipeline(file, originalUrl, onProgress);
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

async function runOfflineStubPipeline(
  file: File,
  originalUrl: string,
  onProgress?: ProgressCallback,
): Promise<{ detection: DetectionResult }> {
  onProgress?.("detect");
  await pause(280);
  onProgress?.("classify");
  await pause(220);
  onProgress?.("validate");
  await pause(180);

  const preview = (await createImagePreview(file)) ?? originalUrl;
  onProgress?.("analyze");

  const detection: DetectionResult = {
    jobId: `offline-${Date.now()}`,
    confidence: 0,
    originalUrl,
    warpedUrl: preview,
    overlayUrl: preview,
    corners: [],
    fen: START_FEN,
    interactiveFen: START_FEN,
    fenValid: true,
    boardReady: true,
    orientation: "white",
    originalWidth: 0,
    originalHeight: 0,
    outputWidth: 0,
    outputHeight: 0,
    squares: [],
    processingMs: 0,
    metadata: {
      offline: true,
      visionPending: true,
      note: "Enable VITE_VISION_LOCAL=true for on-device YOLO.",
    },
  };

  return { detection };
}

function pause(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export { START_FEN };
