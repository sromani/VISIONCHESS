import * as ort from "onnxruntime-web/wasm";

import { configureOrtWasmEnv } from "../ortWasmEnv";
import { fetchArrayBuffer, workerAssetUrl } from "../workerAssets";

export const LAPS_MODEL_PATH = "/models/laps_model.onnx";
const LAPS_MODEL_URL = workerAssetUrl("models/laps_model.onnx");
const ANALYSIS_RADIUS = 10;

/* eslint-disable @typescript-eslint/no-explicit-any */
type CvModule = any;
type CvMat = any;

let lapsSession: ort.InferenceSession | null = null;
let lapsSessionPromise: Promise<ort.InferenceSession> | null = null;

export async function ensureLapsSession(): Promise<ort.InferenceSession> {
  if (lapsSession) return lapsSession;
  if (!lapsSessionPromise) {
    configureOrtWasmEnv(ort);
    lapsSessionPromise = (async () => {
      const buffer = await fetchArrayBuffer(LAPS_MODEL_URL, "LAPS model");
      return ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
      });
    })();
  }
  lapsSession = await lapsSessionPromise;
  return lapsSession;
}

/** Port of laps.__is_lattice_point (geometric + ONNX fallback). */
export async function isLatticePoint(cv: CvModule, patchBgr: CvMat): Promise<boolean> {
  const gray = new cv.Mat();
  const thresh = new cv.Mat();
  const edges = new cv.Mat();
  const resized = new cv.Mat();
  const dilated = new cv.Mat();
  const mask = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();

  try {
    cv.cvtColor(patchBgr, gray, cv.COLOR_BGR2GRAY);
    cv.threshold(gray, thresh, 0, 255, cv.THRESH_OTSU);
    cv.Canny(thresh, edges, 0, 255);
    cv.resize(edges, resized, new cv.Size(21, 21), 0, 0, cv.INTER_CUBIC);

    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(resized, dilated, kernel);
    kernel.delete();
    cv.copyMakeBorder(
      dilated,
      mask,
      1,
      1,
      1,
      1,
      cv.BORDER_CONSTANT,
      new cv.Scalar(255, 255, 255, 255),
    );
    cv.bitwise_not(mask, mask);

    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);

    let numRhomboid = 0;
    for (let i = 0; i < contours.size(); i += 1) {
      const cnt = contours.get(i);
      const { radius } = cv.minEnclosingCircle(cnt);
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.1 * peri, true);
      if (approx.rows === 4 && radius < 14) numRhomboid += 1;
      approx.delete();
      cnt.delete();
    }

    if (numRhomboid === 4) return true;

    const input = new Float32Array(21 * 21);
    for (let i = 0; i < 21 * 21; i += 1) {
      input[i] = resized.data[i] > 127 ? 1 : 0;
    }
    return await lapsOnnxPredict(input);
  } finally {
    gray.delete();
    thresh.delete();
    edges.delete();
    resized.delete();
    dilated.delete();
    mask.delete();
    contours.delete();
    hierarchy.delete();
  }
}

let lapsOnnxPromise: Promise<ort.InferenceSession> | null = null;

async function lapsOnnxPredict(flat: Float32Array): Promise<boolean> {
  if (!lapsOnnxPromise) lapsOnnxPromise = ensureLapsSession();
  const sess = await lapsOnnxPromise;
  const tensor = new ort.Tensor("float32", flat, [1, 21, 21, 1]);
  const out = await sess.run({ [sess.inputNames[0]]: tensor });
  const key = Object.keys(out)[0];
  const pred = out[key].data as Float32Array;
  return pred[0] > pred[1] && pred[1] < 0.03 && pred[0] > 0.975;
}

/** Port of laps.check_board_position — score a quad candidate. */
export async function scoreBoardCorners(
  cv: CvModule,
  bgr: CvMat,
  corners: Point2Like[],
  tolerance = 20,
): Promise<{ ok: boolean; hits: number; warped: CvMat }> {
  const warped = warpBoard(cv, bgr, corners, 1200);
  let hits = 0;
  for (let row = 150; row < 1200; row += 150) {
    for (let col = 150; col < 1200; col += 150) {
      const lx1 = Math.max(0, row - ANALYSIS_RADIUS - 1);
      const lx2 = Math.min(1200, row + ANALYSIS_RADIUS);
      const ly1 = Math.max(0, col - ANALYSIS_RADIUS);
      const ly2 = Math.min(1200, col + ANALYSIS_RADIUS + 1);
      if (lx2 <= lx1 || ly2 <= ly1) continue;
      const roi = warped.roi(new cv.Rect(lx1, ly1, lx2 - lx1, ly2 - ly1));
      if (roi.rows > 0 && roi.cols > 0 && (await isLatticePoint(cv, roi))) hits += 1;
      roi.delete();
    }
  }
  return { ok: hits >= tolerance, hits, warped };
}

export interface Point2Like {
  x: number;
  y: number;
}

export function warpBoard(cv: CvModule, bgr: CvMat, corners: Point2Like[], size: number): CvMat {
  const src = cv.matFromArray(4, 1, cv.CV_32FC2, [
    corners[0].x,
    corners[0].y,
    corners[1].x,
    corners[1].y,
    corners[2].x,
    corners[2].y,
    corners[3].x,
    corners[3].y,
  ]);
  const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, size - 1, 0, size - 1, size - 1, 0, size - 1]);
  const M = cv.getPerspectiveTransform(src, dst);
  const out = new cv.Mat();
  cv.warpPerspective(bgr, out, M, new cv.Size(size, size), cv.INTER_LINEAR, cv.BORDER_CONSTANT);
  src.delete();
  dst.delete();
  M.delete();
  return out;
}

export function cropPatchBgr(cv: CvModule, bgr: CvMat, cx: number, cy: number): CvMat | null {
  const lx1 = Math.max(0, Math.floor(cx - ANALYSIS_RADIUS - 1));
  const lx2 = Math.min(bgr.cols, Math.floor(cx + ANALYSIS_RADIUS));
  const ly1 = Math.max(0, Math.floor(cy - ANALYSIS_RADIUS));
  const ly2 = Math.min(bgr.rows, Math.floor(cy + ANALYSIS_RADIUS + 1));
  if (lx2 <= lx1 || ly2 <= ly1) return null;
  return bgr.roi(new cv.Rect(lx1, ly1, lx2 - lx1, ly2 - ly1));
}
