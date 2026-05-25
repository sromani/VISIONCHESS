import type { DetectionResult } from "@/types";
import type { SavedBoardSnapshot } from "@/types/boardSnapshot";

export function createStubDetection(snapshot: SavedBoardSnapshot): DetectionResult {
  const preview = snapshot.imagePreview ?? "";

  return {
    jobId: snapshot.id,
    confidence: 1,
    originalUrl: preview,
    warpedUrl: preview,
    overlayUrl: preview,
    corners: [],
    fen: snapshot.fen,
    interactiveFen: snapshot.fen,
    fenValid: true,
    boardReady: true,
    orientation: snapshot.orientation,
    originalWidth: 0,
    originalHeight: 0,
    outputWidth: 0,
    outputHeight: 0,
    squares: [],
    processingMs: 0,
  };
}
