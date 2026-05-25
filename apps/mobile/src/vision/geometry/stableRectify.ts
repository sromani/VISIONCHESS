/**
 * stable_v1 — simple board localization (pre multi-candidate regression).
 * 1) YOLO board bbox → warp directly (no OpenCV search overwriting YOLO).
 * 2) Else single-pass contour+LAPS, then Hough+LAPS fallback.
 */
import { detectBoardCorners, warpPreviewRgba, warpToRectifiedRgba } from "./boardDetect";
import { ensureLapsSession } from "./lapsDetect";
import type { Point2, RectifyMetrics, RectifyResult } from "./types";
import type { ImageRGBA } from "../yolo/types";

/* eslint-disable @typescript-eslint/no-explicit-any */
type CvModule = any;
type CvMat = any;

export interface StableRectifyInput {
  yoloCorners?: Point2[];
  yoloConfidence?: number;
}

function emptyResult(image: ImageRGBA, elapsed: number, reason: string): RectifyResult {
  return {
    corners: [],
    rectified: image,
    warpedPreview: image,
    metrics: {
      geometryMs: elapsed,
      method: "fallback_full_frame",
      localizationScore: 0,
      candidateCount: 0,
    },
    geometryOk: false,
    boardFound: false,
    localizationStatus: "not_found",
    localizationDebug: {
      candidates: [],
      attempts: 0,
      bestSource: null,
      rejectionReason: reason,
    },
  };
}

function warpResult(
  cv: CvModule,
  bgr: CvMat,
  image: ImageRGBA,
  corners: Point2[],
  method: RectifyResult["metrics"]["method"],
  extra: {
    lapsHits?: number;
    boardConfidence?: number;
    localizationScore: number;
    boardFound: boolean;
    status: "ok" | "weak";
    source: string;
  },
  elapsed: number,
): RectifyResult {
  const previewSize = Math.max(image.width, image.height, 512);
  const warpedPreview = warpPreviewRgba(cv, bgr, corners, previewSize);
  const rectified = warpToRectifiedRgba(cv, bgr, corners);

  return {
    corners,
    rectified: {
      width: rectified.width,
      height: rectified.height,
      data: rectified.data,
    },
    warpedPreview: {
      width: warpedPreview.width,
      height: warpedPreview.height,
      data: warpedPreview.data,
    },
    metrics: {
      geometryMs: elapsed,
      method,
      lapsHits: extra.lapsHits,
      boardConfidence: extra.boardConfidence,
      localizationScore: extra.localizationScore,
      candidateCount: 1,
    },
    geometryOk: true,
    boardFound: extra.boardFound,
    localizationStatus: extra.status,
    localizationDebug: {
      candidates: [
        {
          source: extra.source,
          score: extra.localizationScore,
          lapsHits: extra.lapsHits ?? 0,
          lapsOk: (extra.lapsHits ?? 0) >= 15,
          geometry: 0,
          grid: 0,
          selected: true,
        },
      ],
      attempts: 1,
      bestSource: extra.source,
      rejectionReason: null,
    },
  };
}

export async function rectifyStableV1(
  cv: CvModule,
  image: ImageRGBA,
  bgr: CvMat,
  input: StableRectifyInput,
): Promise<RectifyResult> {
  const t0 = performance.now();
  await ensureLapsSession();

  if (input.yoloCorners?.length === 4) {
    const conf = input.yoloConfidence ?? 0;
    const elapsed = performance.now() - t0;
    return warpResult(cv, bgr, image, input.yoloCorners, "yolo_board_bbox", {
      lapsHits: undefined,
      boardConfidence: conf,
      localizationScore: Math.min(1, 0.5 + conf * 0.5),
      boardFound: conf >= 0.15,
      status: conf >= 0.25 ? "ok" : "weak",
      source: "yolo_board_bbox_direct",
    }, elapsed);
  }

  const detected = await detectBoardCorners(cv, bgr);
  const elapsed = performance.now() - t0;

  if (!detected || detected.corners.length !== 4) {
    return emptyResult(image, elapsed, "contour_and_hough_failed");
  }

  const lapsOk = detected.lapsHits >= 15;
  const method = detected.method as RectifyMetrics["method"];
  return warpResult(cv, bgr, image, detected.corners, method, {
    lapsHits: detected.lapsHits,
    localizationScore: Math.min(1, detected.lapsHits / 49),
    boardFound: lapsOk,
    status: lapsOk ? "ok" : "weak",
    source: detected.method,
  }, elapsed);
}
