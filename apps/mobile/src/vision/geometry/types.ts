import type { ImageRGBA } from "../yolo/types";

export interface Point2 {
  x: number;
  y: number;
}

export interface LocalizationCandidateDebug {
  source: string;
  score: number;
  lapsHits: number;
  lapsOk: boolean;
  geometry: number;
  grid: number;
  selected: boolean;
}

export interface LocalizationDebug {
  candidates: LocalizationCandidateDebug[];
  attempts: number;
  bestSource: string | null;
  rejectionReason?: string | null;
}

export interface RectifyMetrics {
  geometryMs: number;
  method:
    | "yolo_board_bbox"
    | "laps_scored_quad"
    | "laps_hough"
    | "fallback_full_frame";
  lapsHits?: number;
  boardConfidence?: number;
  localizationScore?: number;
  candidateCount?: number;
}

/** ok = confiable; weak = hay quad pero revisar; not_found = sin candidato usable */
export type LocalizationStatus = "ok" | "weak" | "not_found";

export interface RectifyResult {
  /** Corners in original image coords: TL, TR, BR, BL */
  corners: Point2[];
  rectified: ImageRGBA;
  warpedPreview: ImageRGBA;
  metrics: RectifyMetrics;
  /** True when warp was applied (4 corners). */
  geometryOk: boolean;
  /** Whether we believe a chessboard was found (vs random quad / full frame). */
  boardFound: boolean;
  localizationStatus: LocalizationStatus;
  localizationDebug?: LocalizationDebug;
}

export interface GeometryWorkerIn {
  type: "init" | "rectify";
  id?: number;
  image?: ImageRGBA;
  corners?: Point2[];
  yoloConfidence?: number;
  detectionMode?: string;
}

export interface GeometryWorkerOut {
  type: "ready" | "result" | "error";
  id?: number;
  result?: RectifyResult;
  message?: string;
}
