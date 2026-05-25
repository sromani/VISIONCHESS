/** experimental — multi-candidate localization (kept for A/B, not default). */
import { warpPreviewRgba, warpToRectifiedRgba } from "./boardDetect";
import { localizeBoard, type YoloBoardHint } from "./boardLocalization";
import type { Point2, RectifyResult } from "./types";
import type { ImageRGBA } from "../yolo/types";

/* eslint-disable @typescript-eslint/no-explicit-any */
type CvModule = any;
type CvMat = any;

export async function rectifyExperimental(
  cv: CvModule,
  image: ImageRGBA,
  bgr: CvMat,
  yoloCorners?: Point2[],
  yoloConfidence?: number,
): Promise<RectifyResult> {
  const t0 = performance.now();
  const hints: YoloBoardHint[] = [];
  if (yoloCorners?.length === 4) {
    hints.push({ corners: yoloCorners, confidence: yoloConfidence ?? 0.5 });
  }

  const localized = await localizeBoard(cv, bgr, image.width, image.height, hints);
  const elapsed = performance.now() - t0;

  if (!localized) {
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
        rejectionReason: "no_candidates",
      },
    };
  }

  const previewSize = Math.max(image.width, image.height, 512);
  const warpedPreview = warpPreviewRgba(cv, bgr, localized.corners, previewSize);
  const rectified = warpToRectifiedRgba(cv, bgr, localized.corners);

  const method = localized.method.includes("yolo")
    ? "yolo_board_bbox"
    : localized.method.startsWith("hough")
      ? "laps_hough"
      : "laps_scored_quad";

  return {
    corners: localized.corners,
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
      method: method as RectifyResult["metrics"]["method"],
      lapsHits: localized.lapsHits,
      boardConfidence: localized.score.yoloBoost > 0 ? yoloConfidence : undefined,
      localizationScore: localized.score.total,
      candidateCount: localized.debug.candidates.length,
    },
    geometryOk: true,
    boardFound: localized.boardFound,
    localizationStatus: localized.localizationStatus,
    localizationDebug: localized.debug,
  };
}
