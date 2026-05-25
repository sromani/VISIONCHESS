import { OFFLINE_MODE } from "@/lib/config";
import { detectBoard } from "@/lib/api/detectBoard";
import { ApiError } from "@/lib/api/client";
import { createImagePreview } from "@/lib/storage/imagePreview";
import type { DetectionResult } from "@/types";

export type ProgressCallback = (
  step: "upload" | "detect" | "classify" | "validate" | "analyze",
) => void;

const START_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export async function runVisionPipeline(
  file: File,
  onProgress?: ProgressCallback,
): Promise<{ detection: DetectionResult }> {
  const originalUrl = URL.createObjectURL(file);

  onProgress?.("upload");

  if (!OFFLINE_MODE) {
    return runRemotePipeline(file, originalUrl, onProgress);
  }

  return runOfflinePipeline(file, originalUrl, onProgress);
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

/** Offline placeholder — Phase 4 will swap in OpenCV/YOLO workers. */
async function runOfflinePipeline(
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
      note: "On-device board detection (OpenCV + YOLO) will replace this stub.",
    },
  };

  return { detection };
}

function pause(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export { START_FEN };
