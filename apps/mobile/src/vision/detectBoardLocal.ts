import { detectBoardFromFile } from "./detectBoardPipeline";

/** Offline board detect — default mode: stable_v1 (see VITE_DETECTION_MODE). */
export async function detectBoardLocal(file: File) {
  return detectBoardFromFile({ file });
}

export { detectBoardFromFile } from "./detectBoardPipeline";
export { getDetectionMode, DETECTION_MODES } from "./detection/detectionMode";
export { runYoloInWorker } from "./visionClient";
export type { LocalYoloResult } from "./yolo/types";
