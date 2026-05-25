import {
  candidateKey,
  scoreBoardCandidate,
  type BoardCandidateScore,
  type ScoredBoardCandidate,
} from "./boardCandidateScore";
import {
  collectContourQuads,
  collectHoughQuads,
  quadFromExpandedBbox,
  quadFromLargestQuadContour,
  type RawBoardCandidate,
} from "./boardDetect";
import { orderPointsYSort } from "./orderPoints";
import type { Point2 } from "./types";

/* eslint-disable @typescript-eslint/no-explicit-any */
type CvModule = any;
type CvMat = any;

export interface YoloBoardHint {
  corners: Point2[];
  confidence: number;
  areaRatio?: number;
}

export interface BoardLocalizationDebug {
  candidates: {
    source: string;
    score: number;
    lapsHits: number;
    lapsOk: boolean;
    geometry: number;
    grid: number;
    selected: boolean;
  }[];
  attempts: number;
  bestSource: string | null;
}

export interface BoardLocalizationResult {
  corners: Point2[];
  method: string;
  score: BoardCandidateScore;
  lapsHits: number;
  geometryOk: boolean;
  boardFound: boolean;
  localizationStatus: "ok" | "weak";
  debug: BoardLocalizationDebug;
}

const CONTOUR_PRESETS = [
  { minArea: 0.015, maxArea: 0.99, cannyLow: 25, cannyHigh: 75, epsilon: 0.015 },
  { minArea: 0.02, maxArea: 0.98, cannyLow: 30, cannyHigh: 90, epsilon: 0.02 },
  { minArea: 0.03, maxArea: 0.98, cannyLow: 40, cannyHigh: 120, epsilon: 0.02 },
  { minArea: 0.02, maxArea: 0.98, cannyLow: 50, cannyHigh: 150, epsilon: 0.03 },
  { minArea: 0.025, maxArea: 0.95, cannyLow: 35, cannyHigh: 100, epsilon: 0.025 },
];

const HOUGH_HEIGHTS = [420, 520, 640];

const YOLO_PADDING = [0, 0.03, 0.06, 0.1, 0.14];

/** High-confidence localization. */
const ACCEPT_SCORE = 0.32;
/** Still counts as "board found" for UI. */
const BOARD_FOUND_SCORE = 0.1;
const BOARD_FOUND_LAPS = 6;

function dedupeCandidates(raw: RawBoardCandidate[]): RawBoardCandidate[] {
  const seen = new Set<string>();
  const out: RawBoardCandidate[] = [];
  for (const c of raw) {
    const key = candidateKey(c.corners);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(c);
  }
  return out;
}

async function generateAllCandidates(
  cv: CvModule,
  bgr: CvMat,
  width: number,
  height: number,
  yoloHints: YoloBoardHint[],
): Promise<{ candidates: RawBoardCandidate[]; attempts: number }> {
  const all: RawBoardCandidate[] = [];
  let attempts = 0;

  for (const hint of yoloHints) {
    for (const pad of YOLO_PADDING) {
      attempts += 1;
      const expanded = quadFromExpandedBbox(hint.corners, width, height, pad);
      all.push({
        corners: expanded,
        source: `yolo_bbox_pad_${Math.round(pad * 100)}`,
        yoloConfidence: hint.confidence,
      });
      attempts += 1;
      const refined = await collectContourQuads(cv, bgr, {
        minArea: 0.02,
        maxArea: 0.99,
        cannyLow: 35,
        cannyHigh: 110,
        epsilon: 0.02,
        roiCorners: expanded,
        roiExpand: 0.12,
      });
      all.push(...refined.map((c) => ({ ...c, source: `yolo_roi_refine_${c.source}` })));
    }
  }

  for (const preset of CONTOUR_PRESETS) {
    attempts += 1;
    const quads = await collectContourQuads(cv, bgr, preset);
    all.push(...quads);
  }

  for (const h of HOUGH_HEIGHTS) {
    attempts += 1;
    const quads = await collectHoughQuads(cv, bgr, h);
    all.push(...quads);
  }

  attempts += 1;
  const largest = quadFromLargestQuadContour(cv, bgr);
  if (largest) all.push(largest);

  return { candidates: dedupeCandidates(all), attempts };
}

/** Full-image board search: many candidates, score, pick best (never assume centered). */
export async function localizeBoard(
  cv: CvModule,
  bgr: CvMat,
  imageWidth: number,
  imageHeight: number,
  yoloHints: YoloBoardHint[] = [],
): Promise<BoardLocalizationResult | null> {
  const { candidates: raw, attempts } = await generateAllCandidates(
    cv,
    bgr,
    imageWidth,
    imageHeight,
    yoloHints,
  );

  if (raw.length === 0) {
    if (yoloHints.length > 0) {
      const hint = yoloHints[0];
      const score = await scoreBoardCandidate(cv, bgr, hint.corners, imageWidth, imageHeight, {
        lapsTolerance: 10,
        yoloConfidence: hint.confidence,
      });
      return buildResult(hint.corners, "yolo_bbox_fallback", score, attempts, "weak", true, [
        { corners: orderPointsYSort(hint.corners), source: "yolo_bbox_fallback", score },
      ]);
    }
    return null;
  }

  const scored: ScoredBoardCandidate[] = [];
  for (const c of raw) {
    const score = await scoreBoardCandidate(cv, bgr, c.corners, imageWidth, imageHeight, {
      lapsTolerance: c.source.startsWith("yolo") ? 10 : 12,
      yoloConfidence: c.yoloConfidence,
    });
    scored.push({ corners: orderPointsYSort(c.corners), source: c.source, score });
  }

  scored.sort((a, b) => b.score.total - a.score.total);

  let best = scored[0];
  if (!best) return null;

  const yoloBest = scored.find((c) => c.source.includes("yolo") && (c.score.yoloBoost > 0 || c.source.includes("yolo")));
  if (yoloBest && best.score.total - yoloBest.score.total < 0.12 && yoloBest.score.total >= BOARD_FOUND_SCORE) {
    best = yoloBest;
  }

  const boardFound =
    best.score.total >= BOARD_FOUND_SCORE ||
    best.score.lapsHits >= BOARD_FOUND_LAPS ||
    best.score.lapsOk ||
    best.source.includes("yolo");

  const status: "ok" | "weak" =
    best.score.total >= ACCEPT_SCORE || best.score.lapsOk ? "ok" : "weak";

  return buildResult(best.corners, best.source, best.score, attempts, status, boardFound, scored);
}

function buildResult(
  corners: Point2[],
  source: string,
  score: BoardCandidateScore,
  attempts: number,
  status: "ok" | "weak",
  boardFound: boolean,
  scored?: ScoredBoardCandidate[],
): BoardLocalizationResult {
  const debug: BoardLocalizationDebug = {
    attempts,
    bestSource: source,
    candidates: (scored ?? []).slice(0, 12).map((c, i) => ({
      source: c.source,
      score: Math.round(c.score.total * 1000) / 1000,
      lapsHits: c.score.lapsHits,
      lapsOk: c.score.lapsOk,
      geometry: Math.round(c.score.geometry * 100) / 100,
      grid: Math.round(c.score.grid * 100) / 100,
      selected: i === 0,
    })),
  };

  return {
    corners: orderPointsYSort(corners),
    method: source,
    score,
    lapsHits: score.lapsHits,
    geometryOk: true,
    boardFound,
    localizationStatus: status,
    debug,
  };
}

export { ACCEPT_SCORE, BOARD_FOUND_SCORE };
