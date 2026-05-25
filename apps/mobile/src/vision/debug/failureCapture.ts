import type { DetectionResult } from "@/types";
import type { RectifyResult } from "@/vision/geometry/types";
import type { DetectionMode } from "@/vision/detection/detectionMode";
import type { ImageRGBA } from "@/vision/yolo/types";

const STORAGE_KEY = "visionchess_debug_failures";
const MAX_STORED = 30;

export interface FailureCaseRecord {
  id: string;
  timestamp: string;
  mode: DetectionMode;
  reason: string;
  fileName?: string;
  localizationScore?: number;
  boardFound?: boolean;
  pieceCount?: number;
  bestSource?: string | null;
  rejectionReason?: string | null;
  candidates?: unknown;
  originalDataUrl?: string;
  overlayDataUrl?: string;
}

function loadRecords(): FailureCaseRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as FailureCaseRecord[];
  } catch {
    return [];
  }
}

function saveRecords(records: FailureCaseRecord[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records.slice(0, MAX_STORED)));
  } catch {
    /* quota */
  }
}

export function imageToDataUrl(image: ImageRGBA): string {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.putImageData(
    new ImageData(new Uint8ClampedArray(image.data), image.width, image.height),
    0,
    0,
  );
  return canvas.toDataURL("image/jpeg", 0.85);
}

/** Persist failed scan for offline debug (export via Benchmark panel). */
export function captureFailureCase(params: {
  mode: DetectionMode;
  reason: string;
  fileName?: string;
  geometry: RectifyResult;
  detection?: DetectionResult;
  originalImage?: ImageRGBA;
  overlayDataUrl?: string;
}): FailureCaseRecord {
  const record: FailureCaseRecord = {
    id: `fail-${Date.now()}`,
    timestamp: new Date().toISOString(),
    mode: params.mode,
    reason: params.reason,
    fileName: params.fileName,
    localizationScore: params.geometry.metrics.localizationScore,
    boardFound: params.geometry.boardFound,
    pieceCount: params.detection?.metadata?.yolo_piece_boxes as number | undefined,
    bestSource: params.geometry.localizationDebug?.bestSource ?? null,
    rejectionReason: params.geometry.localizationDebug?.rejectionReason ?? null,
    candidates: params.geometry.localizationDebug?.candidates,
    originalDataUrl: params.originalImage
      ? imageToDataUrl(params.originalImage)
      : params.detection?.originalUrl,
    overlayDataUrl: params.overlayDataUrl ?? params.detection?.overlayUrl,
  };

  const records = [record, ...loadRecords()];
  saveRecords(records);
  console.warn("[VisionChess] debug failure captured", record.id, record.reason);
  return record;
}

export function listFailureCases(): FailureCaseRecord[] {
  return loadRecords();
}

export function clearFailureCases(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function downloadFailureExport(): void {
  const records = loadRecords();
  const blob = new Blob([JSON.stringify(records, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `debug_failures_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  console.info(
    "[VisionChess] Unpack to debug_failures/: node scripts/unpack-debug-failures.cjs <downloaded.json>",
  );
}
