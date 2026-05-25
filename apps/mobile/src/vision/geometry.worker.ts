/**
 * Board localization worker — mode: stable_v1 (default) | experimental | yolo_v1
 */
import cvModule from "@techstark/opencv-js";

import { rectifyExperimental } from "./geometry/experimentalRectify";
import { rectifyStableV1 } from "./geometry/stableRectify";
import { warpPreviewRgba, warpToRectifiedRgba } from "./geometry/boardDetect";
import type { DetectionMode } from "./detection/detectionMode";
import type { GeometryWorkerIn, GeometryWorkerOut, Point2, RectifyResult } from "./geometry/types";
import type { ImageRGBA } from "./yolo/types";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const cv: any = cvModule;

let cvReady: Promise<void> | null = null;

const CV_INIT_TIMEOUT_MS = 45_000;

function waitForCv(): Promise<void> {
  if (cvReady) return cvReady;
  cvReady = new Promise((resolve, reject) => {
    if (cv.Mat) {
      resolve();
      return;
    }
    const deadline = performance.now() + CV_INIT_TIMEOUT_MS;
    cv.onRuntimeInitialized = () => resolve();
    const poll = () => {
      if (cv.Mat) {
        resolve();
        return;
      }
      if (performance.now() > deadline) {
        reject(new Error("OpenCV runtime init timeout"));
        return;
      }
      setTimeout(poll, 50);
    };
    poll();
  });
  return cvReady;
}

function rgbaToBgr(image: ImageRGBA): ReturnType<typeof cv.Mat> {
  const rgba = cv.matFromImageData({
    data: image.data,
    width: image.width,
    height: image.height,
  });
  const bgr = new cv.Mat();
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
  rgba.delete();
  return bgr;
}

/** yolo_v1: only trust YOLO board class — no OpenCV overwrite. */
async function rectifyYoloV1(
  image: ImageRGBA,
  bgr: ReturnType<typeof cv.Mat>,
  yoloCorners?: Point2[],
  yoloConfidence?: number,
): Promise<RectifyResult> {
  const t0 = performance.now();
  if (yoloCorners?.length !== 4) {
    return {
      corners: [],
      rectified: image,
      warpedPreview: image,
      metrics: { geometryMs: performance.now() - t0, method: "fallback_full_frame" },
      geometryOk: false,
      boardFound: false,
      localizationStatus: "not_found",
      localizationDebug: {
        candidates: [],
        attempts: 0,
        bestSource: null,
        rejectionReason: "yolo_board_not_detected",
      },
    };
  }

  const previewSize = Math.max(image.width, image.height, 512);
  const warpedPreview = warpPreviewRgba(cv, bgr, yoloCorners, previewSize);
  const rectified = warpToRectifiedRgba(cv, bgr, yoloCorners);
  const conf = yoloConfidence ?? 0;
  const elapsed = performance.now() - t0;

  return {
    corners: yoloCorners,
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
      method: "yolo_board_bbox",
      boardConfidence: conf,
      localizationScore: conf,
      candidateCount: 1,
    },
    geometryOk: true,
    boardFound: conf >= 0.15,
    localizationStatus: conf >= 0.25 ? "ok" : "weak",
    localizationDebug: {
      candidates: [
        {
          source: "yolo_only",
          score: conf,
          lapsHits: 0,
          lapsOk: false,
          geometry: 0,
          grid: 0,
          selected: true,
        },
      ],
      attempts: 1,
      bestSource: "yolo_only",
      rejectionReason: null,
    },
  };
}

async function rectifyImage(
  image: ImageRGBA,
  mode: DetectionMode,
  yoloCorners?: Point2[],
  yoloConfidence?: number,
): Promise<RectifyResult> {
  await waitForCv();
  const bgr = rgbaToBgr(image);

  try {
    if (mode === "experimental") {
      return await rectifyExperimental(cv, image, bgr, yoloCorners, yoloConfidence);
    }
    if (mode === "yolo_v1") {
      return await rectifyYoloV1(image, bgr, yoloCorners, yoloConfidence);
    }
    // mesh_v2 → stable until mesh port exists
    return await rectifyStableV1(cv, image, bgr, {
      yoloCorners,
      yoloConfidence,
    });
  } finally {
    bgr.delete();
  }
}

self.onmessage = (ev: MessageEvent<GeometryWorkerIn>) => {
  const msg = ev.data;
  void (async () => {
    try {
      if (msg.type === "init") {
        await waitForCv();
        // LAPS loads on first rectify (avoid blocking app bootstrap).
        self.postMessage({ type: "ready" } satisfies GeometryWorkerOut);
        return;
      }

      if (msg.type === "rectify" && msg.image && msg.id != null) {
        const mode = (msg.detectionMode ?? "stable_v1") as DetectionMode;
        const result = await rectifyImage(
          msg.image,
          mode,
          msg.corners,
          msg.yoloConfidence,
        );
        self.postMessage({
          type: "result",
          id: msg.id,
          result,
        } satisfies GeometryWorkerOut);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[Geometry worker]", err);
      self.postMessage({
        type: "error",
        id: msg.type === "rectify" ? msg.id : undefined,
        message,
      } satisfies GeometryWorkerOut);
    }
  })();
};
