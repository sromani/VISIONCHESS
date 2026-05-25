import { scoreBoardCorners } from "./lapsDetect";
import { orderPointsYSort } from "./orderPoints";
import type { Point2 } from "./types";

/* eslint-disable @typescript-eslint/no-explicit-any */
type CvModule = any;
type CvMat = any;

export interface BoardCandidateScore {
  total: number;
  lapsHits: number;
  lapsOk: boolean;
  geometry: number;
  grid: number;
  areaFit: number;
  yoloBoost: number;
}

export interface ScoredBoardCandidate {
  corners: Point2[];
  source: string;
  score: BoardCandidateScore;
}

function dist(a: Point2, b: Point2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Convex quad sanity: side lengths, aspect ratio vs image, corner angles. */
export function scoreQuadGeometry(
  corners: Point2[],
  imageWidth: number,
  imageHeight: number,
): number {
  if (corners.length !== 4) return 0;
  const ordered = orderPointsYSort(corners);
  const wTop = dist(ordered[0], ordered[1]);
  const wBot = dist(ordered[3], ordered[2]);
  const hLeft = dist(ordered[0], ordered[3]);
  const hRight = dist(ordered[1], ordered[2]);
  const w = (wTop + wBot) / 2;
  const h = (hLeft + hRight) / 2;
  if (w < 8 || h < 8) return 0;

  const aspect = w / h;
  const aspectScore = 1 - Math.min(1, Math.abs(aspect - 1) / 0.55);

  const imgDiag = Math.hypot(imageWidth, imageHeight);
  const sizeScore = Math.min(1, (w * h) / (imgDiag * imgDiag * 0.12));

  const sideRatio = Math.min(wTop, wBot, hLeft, hRight) / Math.max(wTop, wBot, hLeft, hRight);
  const parallelScore = sideRatio;

  const cosScores: number[] = [];
  for (let i = 0; i < 4; i += 1) {
    const p0 = ordered[(i + 3) % 4];
    const p1 = ordered[i];
    const p2 = ordered[(i + 1) % 4];
    const v1x = p0.x - p1.x;
    const v1y = p0.y - p1.y;
    const v2x = p2.x - p1.x;
    const v2y = p2.y - p1.y;
    const n1 = Math.hypot(v1x, v1y) || 1;
    const n2 = Math.hypot(v2x, v2y) || 1;
    const cos = Math.abs((v1x * v2x + v1y * v2y) / (n1 * n2));
    cosScores.push(1 - Math.min(1, cos / 0.35));
  }
  const angleScore = cosScores.reduce((a, b) => a + b, 0) / 4;

  return 0.3 * aspectScore + 0.25 * sizeScore + 0.2 * parallelScore + 0.25 * angleScore;
}

/** After warp: edge energy peaks along 7 internal grid lines (8×8 board). */
export function scoreGridRegularity(cv: CvModule, bgr: CvMat, corners: Point2[]): number {
  const size = 400;
  const warped = warpSmall(cv, bgr, corners, size);
  const gray = new cv.Mat();
  const edges = new cv.Mat();
  try {
    cv.cvtColor(warped, gray, cv.COLOR_BGR2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(3, 3), 0);
    cv.Canny(gray, edges, 40, 120);

    const rowProj = new Float32Array(size).fill(0);
    const colProj = new Float32Array(size).fill(0);
    const data = edges.data;
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        if (data[y * size + x] > 0) {
          rowProj[y] += 1;
          colProj[x] += 1;
        }
      }
    }

    const rowPeaks = countNearUniformPeaks(rowProj, 8, size * 0.04);
    const colPeaks = countNearUniformPeaks(colProj, 8, size * 0.04);
    const peakScore = (rowPeaks + colPeaks) / 16;

    const rowVar = varianceScore(rowProj, 8);
    const colVar = varianceScore(colProj, 8);
    return Math.min(1, 0.55 * peakScore + 0.225 * rowVar + 0.225 * colVar);
  } finally {
    warped.delete();
    gray.delete();
    edges.delete();
  }
}

function warpSmall(cv: CvModule, bgr: CvMat, corners: Point2[], size: number): CvMat {
  const ordered = orderPointsYSort(corners);
  const src = cv.matFromArray(4, 1, cv.CV_32FC2, [
    ordered[0].x, ordered[0].y, ordered[1].x, ordered[1].y,
    ordered[2].x, ordered[2].y, ordered[3].x, ordered[3].y,
  ]);
  const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0, size - 1, 0, size - 1, size - 1, 0, size - 1,
  ]);
  const M = cv.getPerspectiveTransform(src, dst);
  const out = new cv.Mat();
  cv.warpPerspective(bgr, out, M, new cv.Size(size, size), cv.INTER_LINEAR, cv.BORDER_CONSTANT);
  src.delete();
  dst.delete();
  M.delete();
  return out;
}

function countNearUniformPeaks(proj: Float32Array, expected: number, band: number): number {
  const len = proj.length;
  const max = Math.max(...proj);
  if (max < 1) return 0;
  const threshold = max * 0.35;
  const peaks: number[] = [];
  for (let i = 1; i < len - 1; i += 1) {
    if (proj[i] >= threshold && proj[i] >= proj[i - 1] && proj[i] >= proj[i + 1]) {
      peaks.push(i);
    }
  }
  if (peaks.length < 4) return peaks.length / expected;

  const step = len / (expected + 1);
  let matched = 0;
  for (let k = 1; k <= expected; k += 1) {
    const target = step * k;
    if (peaks.some((p) => Math.abs(p - target) <= band)) matched += 1;
  }
  return matched / expected;
}

function varianceScore(proj: Float32Array, bins: number): number {
  const len = proj.length;
  const binSize = len / bins;
  const means: number[] = [];
  for (let b = 0; b < bins; b += 1) {
    let sum = 0;
    const start = Math.floor(b * binSize);
    const end = Math.floor((b + 1) * binSize);
    for (let i = start; i < end; i += 1) sum += proj[i];
    means.push(sum / Math.max(1, end - start));
  }
  const avg = means.reduce((a, c) => a + c, 0) / means.length;
  if (avg < 1) return 0;
  const cv = Math.sqrt(means.reduce((s, m) => s + (m - avg) ** 2, 0) / means.length) / avg;
  return Math.min(1, cv / 0.85);
}

export function scoreAreaFit(corners: Point2[], imageWidth: number, imageHeight: number): number {
  const ordered = orderPointsYSort(corners);
  const wTop = dist(ordered[0], ordered[1]);
  const wBot = dist(ordered[3], ordered[2]);
  const hLeft = dist(ordered[0], ordered[3]);
  const hRight = dist(ordered[1], ordered[2]);
  const area = ((wTop + wBot) / 2) * ((hLeft + hRight) / 2);
  const ratio = area / (imageWidth * imageHeight);
  if (ratio < 0.015 || ratio > 0.98) return 0;
  if (ratio >= 0.04 && ratio <= 0.85) return 1;
  if (ratio < 0.04) return ratio / 0.04;
  return Math.max(0, 1 - (ratio - 0.85) / 0.13);
}

export async function scoreBoardCandidate(
  cv: CvModule,
  bgr: CvMat,
  corners: Point2[],
  imageWidth: number,
  imageHeight: number,
  options?: { lapsTolerance?: number; yoloConfidence?: number },
): Promise<BoardCandidateScore> {
  const lapsTolerance = options?.lapsTolerance ?? 12;
  const ordered = orderPointsYSort(corners);
  const { ok, hits, warped } = await scoreBoardCorners(cv, bgr, ordered, lapsTolerance);
  warped.delete();

  const geometry = scoreQuadGeometry(ordered, imageWidth, imageHeight);
  const grid = scoreGridRegularity(cv, bgr, ordered);
  const areaFit = scoreAreaFit(ordered, imageWidth, imageHeight);
  const yoloBoost = options?.yoloConfidence
    ? Math.min(0.15, options.yoloConfidence * 0.12)
    : 0;

  const lapsNorm = Math.min(1, hits / 49);
  const lapsComponent = ok ? 0.35 + 0.25 * lapsNorm : 0.2 * lapsNorm;

  const total = Math.min(
    1,
    lapsComponent + 0.22 * geometry + 0.18 * grid + 0.12 * areaFit + yoloBoost,
  );

  return {
    total,
    lapsHits: hits,
    lapsOk: ok,
    geometry,
    grid,
    areaFit,
    yoloBoost,
  };
}

export function candidateKey(corners: Point2[]): string {
  const o = orderPointsYSort(corners);
  return o.map((p) => `${Math.round(p.x / 8)}_${Math.round(p.y / 8)}`).join("-");
}
