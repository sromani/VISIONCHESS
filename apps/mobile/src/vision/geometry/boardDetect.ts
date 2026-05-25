import { findLineIntersections, clusterPoints, type Segment } from "./lines";
import { isLatticePoint, scoreBoardCorners, warpBoard, cropPatchBgr } from "./lapsDetect";
import { orderPointsYSort } from "./orderPoints";
import type { Point2 } from "./types";

/* eslint-disable @typescript-eslint/no-explicit-any */
type CvModule = any;
type CvMat = any;

const BOARD_SIZE = 1200;

export interface ContourSearchParams {
  minArea: number;
  maxArea: number;
  cannyLow: number;
  cannyHigh: number;
  epsilon: number;
  /** Optional: search only inside expanded quad ROI (full-res coords). */
  roiCorners?: Point2[];
  roiExpand?: number;
}

export interface RawBoardCandidate {
  corners: Point2[];
  source: string;
  yoloConfidence?: number;
}

export interface DetectCornersResult {
  corners: Point2[];
  method: "laps_scored_quad" | "laps_hough" | "yolo_board_bbox" | string;
  lapsHits: number;
}

export function quadFromExpandedBbox(
  corners: Point2[],
  imageWidth: number,
  imageHeight: number,
  padding: number,
): Point2[] {
  const ordered = orderPointsYSort(corners);
  const xs = ordered.map((p) => p.x);
  const ys = ordered.map((p) => p.y);
  const x1 = Math.min(...xs);
  const x2 = Math.max(...xs);
  const y1 = Math.min(...ys);
  const y2 = Math.max(...ys);
  const w = x2 - x1;
  const h = y2 - y1;
  const padX = w * padding;
  const padY = h * padding;
  return orderPointsYSort([
    { x: Math.max(0, x1 - padX), y: Math.max(0, y1 - padY) },
    { x: Math.min(imageWidth, x2 + padX), y: Math.max(0, y1 - padY) },
    { x: Math.min(imageWidth, x2 + padX), y: Math.min(imageHeight, y2 + padY) },
    { x: Math.max(0, x1 - padX), y: Math.min(imageHeight, y2 + padY) },
  ]);
}

function cropRoiMat(
  cv: CvModule,
  bgr: CvMat,
  corners: Point2[],
  expand: number,
): { mat: CvMat; offsetX: number; offsetY: number; scale: number } | null {
  const ordered = orderPointsYSort(corners);
  const xs = ordered.map((p) => p.x);
  const ys = ordered.map((p) => p.y);
  let x1 = Math.min(...xs);
  let x2 = Math.max(...xs);
  let y1 = Math.min(...ys);
  let y2 = Math.max(...ys);
  const w = x2 - x1;
  const h = y2 - y1;
  x1 = Math.max(0, x1 - w * expand);
  y1 = Math.max(0, y1 - h * expand);
  x2 = Math.min(bgr.cols, x2 + w * expand);
  y2 = Math.min(bgr.rows, y2 + h * expand);
  const rw = Math.round(x2 - x1);
  const rh = Math.round(y2 - y1);
  if (rw < 40 || rh < 40) return null;

  const rect = new cv.Rect(Math.round(x1), Math.round(y1), rw, rh);
  const roi = bgr.roi(rect);
  const targetH = 600;
  const scale = targetH / rh;
  const tw = Math.max(1, Math.round(rw * scale));
  const th = targetH;
  const resized = new cv.Mat();
  cv.resize(roi, resized, new cv.Size(tw, th), 0, 0, cv.INTER_AREA);
  roi.delete();
  return { mat: resized, offsetX: x1, offsetY: y1, scale };
}

function mapCornersFromRoi(
  corners: Point2[],
  offsetX: number,
  offsetY: number,
  scale: number,
): Point2[] {
  return corners.map((p) => ({
    x: p.x / scale + offsetX,
    y: p.y / scale + offsetY,
  }));
}

/** Collect all 4-point contour quads (multi-candidate, not first-match). */
export async function collectContourQuads(
  cv: CvModule,
  bgr: CvMat,
  params: ContourSearchParams,
): Promise<RawBoardCandidate[]> {
  let work = bgr;
  let offsetX = 0;
  let offsetY = 0;
  let scale = 1;
  let owned: CvMat | null = null;

  if (params.roiCorners?.length === 4) {
    const roi = cropRoiMat(cv, bgr, params.roiCorners, params.roiExpand ?? 0.1);
    if (!roi) return [];
    work = roi.mat;
    offsetX = roi.offsetX;
    offsetY = roi.offsetY;
    scale = roi.scale;
    owned = work;
  }

  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const out: RawBoardCandidate[] = [];

  try {
    cv.cvtColor(work, gray, cv.COLOR_BGR2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, params.cannyLow, params.cannyHigh);

    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    const imgArea = work.rows * work.cols;
    const minA = imgArea * params.minArea;
    const maxA = imgArea * params.maxArea;

    for (let i = 0; i < contours.size(); i += 1) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area < minA || area > maxA) {
        cnt.delete();
        continue;
      }
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, params.epsilon * peri, true);
      if (approx.rows === 4) {
        const pts: Point2[] = [];
        for (let r = 0; r < 4; r += 1) {
          pts.push({ x: approx.data32S[r * 2], y: approx.data32S[r * 2 + 1] });
        }
        let ordered = orderPointsYSort(pts);
        if (owned) ordered = mapCornersFromRoi(ordered, offsetX, offsetY, scale);
        out.push({ corners: ordered, source: "contour_quad" });
      }
      approx.delete();
      cnt.delete();
    }
  } finally {
    gray.delete();
    blurred.delete();
    edges.delete();
    contours.delete();
    hierarchy.delete();
    owned?.delete();
  }

  return out;
}

/** Legacy single best contour (used by tests / fallback). */
async function findBestContourQuad(cv: CvModule, bgr: CvMat): Promise<DetectCornersResult | null> {
  const quads = await collectContourQuads(cv, bgr, {
    minArea: 0.03,
    maxArea: 0.98,
    cannyLow: 40,
    cannyHigh: 120,
    epsilon: 0.02,
  });
  let best: DetectCornersResult | null = null;
  let bestHits = -1;
  for (const q of quads) {
    const { ok, hits, warped } = await scoreBoardCorners(cv, bgr, q.corners, 18);
    warped.delete();
    if (ok && hits > bestHits) {
      bestHits = hits;
      best = { corners: q.corners, method: "laps_scored_quad", lapsHits: hits };
    }
  }
  return best;
}

export async function collectHoughQuads(
  cv: CvModule,
  bgr: CvMat,
  workHeight: number,
): Promise<RawBoardCandidate[]> {
  const result = await findCornersFromHoughLaps(cv, bgr, workHeight);
  if (!result) return [];
  return [{ corners: result.corners, source: `hough_${workHeight}` }];
}

/** Largest 4-vertex contour by area (square-ish fallback). */
export function quadFromLargestQuadContour(cv: CvModule, bgr: CvMat): RawBoardCandidate | null {
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();

  try {
    cv.cvtColor(bgr, gray, cv.COLOR_BGR2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, 30, 100);
    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    const imgArea = bgr.rows * bgr.cols;
    let bestArea = 0;
    let bestPts: Point2[] | null = null;

    for (let i = 0; i < contours.size(); i += 1) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area < imgArea * 0.02 || area > imgArea * 0.98) {
        cnt.delete();
        continue;
      }
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.02 * peri, true);
      if (approx.rows === 4 && area > bestArea) {
        const pts: Point2[] = [];
        for (let r = 0; r < 4; r += 1) {
          pts.push({ x: approx.data32S[r * 2], y: approx.data32S[r * 2 + 1] });
        }
        bestArea = area;
        bestPts = orderPointsYSort(pts);
      }
      approx.delete();
      cnt.delete();
    }
    if (!bestPts) return null;
    return { corners: bestPts, source: "largest_quad_contour" };
  } finally {
    gray.delete();
    blurred.delete();
    edges.delete();
    contours.delete();
    hierarchy.delete();
  }
}

/** Find board quad: contour candidates scored with LAPS, then Hough+LAPS fallback. */
export async function detectBoardCorners(cv: CvModule, bgr: CvMat): Promise<DetectCornersResult | null> {
  try {
    const contour = await findBestContourQuad(cv, bgr);
    if (contour) return contour;
  } catch (err) {
    console.warn("[boardDetect] contour path failed", err);
  }

  try {
    return await findCornersFromHoughLaps(cv, bgr, 500);
  } catch (err) {
    console.warn("[boardDetect] hough path failed", err);
    return null;
  }
}

function preprocessForLineDetection(cv: CvModule, gray: CvMat, out: CvMat): void {
  const median = new cv.Mat();
  const blurred = new cv.Mat();
  try {
    cv.medianBlur(gray, median, 5);
    cv.GaussianBlur(median, blurred, new cv.Size(7, 7), 2);
    blurred.copyTo(out);
  } finally {
    median.delete();
    blurred.delete();
  }
}

async function findCornersFromHoughLaps(
  cv: CvModule,
  bgr: CvMat,
  workHeight: number,
): Promise<DetectCornersResult | null> {
  const work = resizeToHeight(cv, bgr, workHeight);
  const scale = bgr.rows / work.rows;

  const gray = new cv.Mat();
  const preprocessed = new cv.Mat();
  const edges = new cv.Mat();
  const linesMat = new cv.Mat();

  try {
    cv.cvtColor(work, gray, cv.COLOR_BGR2GRAY);
    preprocessForLineDetection(cv, gray, preprocessed);

    const v = medianValue(preprocessed);
    const lower = Math.max(0, Math.floor(0.75 * v));
    const upper = Math.min(255, Math.floor(1.25 * v));
    cv.Canny(preprocessed, edges, lower, upper);

    cv.HoughLinesP(edges, linesMat, 1, Math.PI / 180, 40, 50, 15);
    const segments: Segment[] = [];
    for (let i = 0; i < linesMat.rows; i += 1) {
      const x1 = linesMat.data32S[i * 4];
      const y1 = linesMat.data32S[i * 4 + 1];
      const x2 = linesMat.data32S[i * 4 + 2];
      const y2 = linesMat.data32S[i * 4 + 3];
      segments.push([[x1, y1], [x2, y2]]);
    }

    const raw = findLineIntersections(segments, 300);
    const clustered = clusterPoints(raw, 12);
    const lattice: [number, number][] = [];
    for (const pt of clustered) {
      const patch = cropPatchBgr(cv, work, pt[0], pt[1]);
      if (!patch) continue;
      if (await isLatticePoint(cv, patch)) lattice.push(pt);
      patch.delete();
    }

    if (lattice.length < 4) return null;

    const corners = cornersFromLattice(cv, lattice);
    if (!corners) return null;

    const scaled = corners.map((p) => ({ x: p.x * scale, y: p.y * scale }));
    const ordered = orderPointsYSort(scaled);
    const { ok, hits, warped } = await scoreBoardCorners(cv, bgr, ordered, 15);
    warped.delete();
    if (!ok) return null;
    return { corners: ordered, method: "laps_hough", lapsHits: hits };
  } finally {
    work.delete();
    gray.delete();
    preprocessed.delete();
    edges.delete();
    linesMat.delete();
  }
}

function cornersFromLattice(cv: CvModule, points: [number, number][]): Point2[] | null {
  const mat = cv.matFromArray(
    points.length,
    1,
    cv.CV_32FC2,
    points.flatMap(([x, y]) => [x, y]),
  );
  const hull = new cv.Mat();
  try {
    cv.convexHull(mat, hull, false, true);
    if (hull.rows < 4) return null;
    const hullPts: [number, number][] = [];
    for (let i = 0; i < hull.rows; i += 1) {
      hullPts.push([hull.data32S[i * 2], hull.data32S[i * 2 + 1]]);
    }
    const sums = hullPts.map(([x, y]) => x + y);
    const diffs = hullPts.map(([x, y]) => x - y);
    const tl = hullPts[sums.indexOf(Math.min(...sums))];
    const br = hullPts[sums.indexOf(Math.max(...sums))];
    const tr = hullPts[diffs.indexOf(Math.max(...diffs))];
    const bl = hullPts[diffs.indexOf(Math.min(...diffs))];
    return [
      { x: tl[0], y: tl[1] },
      { x: tr[0], y: tr[1] },
      { x: br[0], y: br[1] },
      { x: bl[0], y: bl[1] },
    ];
  } finally {
    mat.delete();
    hull.delete();
  }
}

function resizeToHeight(cv: CvModule, src: CvMat, height: number): CvMat {
  const scale = Math.sqrt((height * height) / (src.rows * src.cols));
  const w = Math.max(1, Math.round(src.cols * scale));
  const h = Math.max(1, Math.round(src.rows * scale));
  const out = new cv.Mat();
  cv.resize(src, out, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
  return out;
}

function medianValue(gray: CvMat): number {
  const data = gray.data;
  const samples: number[] = [];
  const step = Math.max(1, Math.floor(data.length / 5000));
  for (let i = 0; i < data.length; i += step) samples.push(data[i]);
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)] ?? 128;
}

export function warpToRectifiedRgba(cv: CvModule, bgr: CvMat, corners: Point2[]): ImageDataLike {
  const warped = warpBoard(cv, bgr, corners, BOARD_SIZE);
  const rgba = new cv.Mat();
  cv.cvtColor(warped, rgba, cv.COLOR_BGR2RGBA);
  const imageData = matToImageData(rgba);
  warped.delete();
  rgba.delete();
  return imageData;
}

export interface ImageDataLike {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

function matToImageData(mat: CvMat): ImageDataLike {
  const width = mat.cols;
  const height = mat.rows;
  const data = new Uint8ClampedArray(width * height * 4);
  data.set(mat.data);
  return { width, height, data };
}

export function warpPreviewRgba(
  cv: CvModule,
  bgr: CvMat,
  corners: Point2[],
  previewSize: number,
): ImageDataLike {
  const warped = warpBoard(cv, bgr, corners, previewSize);
  const rgba = new cv.Mat();
  cv.cvtColor(warped, rgba, cv.COLOR_BGR2RGBA);
  const out = matToImageData(rgba);
  warped.delete();
  rgba.delete();
  return out;
}
