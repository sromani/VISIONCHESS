/**
 * Fast board geometry — pure Canvas/JS, typically &lt;2s on mobile (no OpenCV WASM).
 */
import { orderCorners, type CornerPoint } from "@/lib/vision/cornerOrder";
import type { GeometryTrace, QuadCandidateTrace } from "@/lib/vision/geometryDebugTypes";
import type { WarpQualityPayload } from "@/lib/vision/opencvLoaderTypes";

const MAX_EDGE = 640;

export interface FastRectifyResult {
  corners: CornerPoint[];
  ordered: CornerPoint[];
  homography: number[][];
  warpedRgba: Uint8ClampedArray;
  warpedWidth: number;
  warpedHeight: number;
  trace: GeometryTrace;
  warpQuality: WarpQualityPayload;
  source: ImageData;
}

export async function loadImageDataFromFile(file: File, maxEdge = MAX_EDGE): Promise<ImageData> {
  const bitmap = await createImageBitmap(file);
  let { width, height } = bitmap;
  const scale = Math.min(1, maxEdge / Math.max(width, height));
  if (scale < 1) {
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2d unavailable");
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return ctx.getImageData(0, 0, width, height);
}

function grayscale(data: Uint8ClampedArray, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    out[i] = 0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2];
  }
  return out;
}

function boxBlur3(gray: Float32Array, w: number, h: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      let n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          s += gray[yy * w + xx];
          n++;
        }
      }
      tmp[y * w + x] = s / n;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      let n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          s += tmp[yy * w + xx];
          n++;
        }
      }
      out[y * w + x] = s / n;
    }
  }
  return out;
}

function sobelMag(gray: Float32Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const gx =
        -gray[idx - w - 1] +
        gray[idx - w + 1] +
        -2 * gray[idx - 1] +
        2 * gray[idx + 1] +
        -gray[idx + w - 1] +
        gray[idx + w + 1];
      const gy =
        -gray[idx - w - 1] -
        2 * gray[idx - w] -
        gray[idx - w + 1] +
        gray[idx + w - 1] +
        2 * gray[idx + w] +
        gray[idx + w + 1];
      out[idx] = Math.hypot(gx, gy);
    }
  }
  return out;
}

function percentile(values: Float32Array, p: number): number {
  const arr: number[] = [];
  for (let i = 0; i < values.length; i++) {
    if (values[i] > 0) arr.push(values[i]);
  }
  if (arr.length === 0) return 0;
  arr.sort((a, b) => a - b);
  const idx = Math.min(arr.length - 1, Math.floor((p / 100) * arr.length));
  return arr[idx];
}

function insetCorners(w: number, h: number, margin = 0.12): CornerPoint[] {
  const mx = w * margin;
  const my = h * margin;
  return [
    { x: mx, y: my },
    { x: w - 1 - mx, y: my },
    { x: w - 1 - mx, y: h - 1 - my },
    { x: mx, y: h - 1 - my },
  ];
}

function cornersFromEdges(mag: Float32Array, w: number, h: number): CornerPoint[] | null {
  const thresh = percentile(mag, 72);
  const pts: CornerPoint[] = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mag[y * w + x] >= thresh) pts.push({ x, y });
    }
  }
  if (pts.length < 40) return null;

  const tl = pts.reduce((a, b) => (a.x + a.y < b.x + b.y ? a : b));
  const br = pts.reduce((a, b) => (a.x + a.y > b.x + b.y ? a : b));
  const tr = pts.reduce((a, b) => (a.x - a.y < b.x - b.y ? a : b));
  const bl = pts.reduce((a, b) => (a.x - a.y > b.x - b.y ? a : b));
  return [tl, tr, br, bl];
}

/** 3×3 homography: src quad → axis-aligned square [0,s]² */
function homographyFromQuad(src: CornerPoint[], size: number): number[][] {
  const dst = [
    [0, 0],
    [size - 1, 0],
    [size - 1, size - 1],
    [0, size - 1],
  ];
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = [src[i].x, src[i].y];
    const [xp, yp] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -xp * x, -xp * y]);
    b.push(xp);
    A.push([0, 0, 0, x, y, 1, -yp * x, -yp * y]);
    b.push(yp);
  }
  const h = solve8(A, b);
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

function solve8(A: number[][], b: number[]): number[] {
  const n = 8;
  const m = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const div = m[col][col] || 1e-9;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = m[row][col] / div;
      for (let c = col; c <= n; c++) m[row][c] -= factor * m[col][c];
    }
    for (let c = col; c <= n; c++) m[col][c] /= div;
  }
  return m.map((row) => row[n]);
}

function invert3x3(H: number[][]): number[][] | null {
  const a = H[0][0];
  const b = H[0][1];
  const c = H[0][2];
  const d = H[1][0];
  const e = H[1][1];
  const f = H[1][2];
  const g = H[2][0];
  const h = H[2][1];
  const i = H[2][2];
  const A = e * i - f * h;
  const B = -(b * i - c * h);
  const C = b * f - c * e;
  const D = -(d * i - f * g);
  const E = a * i - c * g;
  const F = -(a * f - c * d);
  const G = d * h - e * g;
  const Hc = -(a * h - b * g);
  const I = a * e - b * d;
  const det = a * A + b * D + c * G;
  if (Math.abs(det) < 1e-12) return null;
  return [
    [A / det, B / det, C / det],
    [D / det, E / det, F / det],
    [G / det, Hc / det, I / det],
  ];
}

function sampleBilinear(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  x: number,
  y: number,
): [number, number, number, number] {
  const x0 = Math.max(0, Math.min(w - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(h - 1, Math.floor(y)));
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;
  const i = (xx: number, yy: number) => (yy * w + xx) * 4;
  const p00 = i(x0, y0);
  const p10 = i(x1, y0);
  const p01 = i(x0, y1);
  const p11 = i(x1, y1);
  const out: [number, number, number, number] = [0, 0, 0, 255];
  for (let c = 0; c < 3; c++) {
    const v00 = data[p00 + c];
    const v10 = data[p10 + c];
    const v01 = data[p01 + c];
    const v11 = data[p11 + c];
    out[c] = Math.round(
      (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11,
    );
  }
  return out;
}

function warpPerspective(
  src: ImageData,
  Hinv: number[][],
  outSize: number,
): Uint8ClampedArray {
  const { width: w, height: h, data } = src;
  const out = new Uint8ClampedArray(outSize * outSize * 4);
  for (let y = 0; y < outSize; y++) {
    for (let x = 0; x < outSize; x++) {
      const xp = Hinv[0][0] * x + Hinv[0][1] * y + Hinv[0][2];
      const yp = Hinv[1][0] * x + Hinv[1][1] * y + Hinv[1][2];
      const wp = Hinv[2][0] * x + Hinv[2][1] * y + Hinv[2][2];
      if (Math.abs(wp) < 1e-9) continue;
      const sx = xp / wp;
      const sy = yp / wp;
      if (sx < 0 || sy < 0 || sx >= w - 1 || sy >= h - 1) continue;
      const rgba = sampleBilinear(data, w, h, sx, sy);
      const o = (y * outSize + x) * 4;
      out[o] = rgba[0];
      out[o + 1] = rgba[1];
      out[o + 2] = rgba[2];
      out[o + 3] = 255;
    }
  }
  return out;
}

function assessWarpQuality(gray: Float32Array, w: number, h: number): WarpQualityPayload {
  let lapVar = 0;
  let contrastStd = 0;
  const cellMeans: number[] = [];
  const cellW = Math.floor(w / 8);
  const cellH = Math.floor(h / 8);
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      let sum = 0;
      let n = 0;
      for (let y = row * cellH; y < (row + 1) * cellH && y < h; y++) {
        for (let x = col * cellW; x < (col + 1) * cellW && x < w; x++) {
          sum += gray[y * w + x];
          n++;
        }
      }
      cellMeans.push(sum / Math.max(1, n));
    }
  }
  const m = cellMeans.reduce((a, b) => a + b, 0) / cellMeans.length;
  const sqStd = Math.sqrt(
    cellMeans.reduce((a, v) => a + (v - m) ** 2, 0) / cellMeans.length,
  );

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const c = gray[y * w + x];
      lapVar += Math.abs(
        -4 * c +
          gray[(y - 1) * w + x] +
          gray[(y + 1) * w + x] +
          gray[y * w + (x - 1)] +
          gray[y * w + (x + 1)],
      );
    }
  }
  lapVar /= (w - 2) * (h - 2);
  let sum = 0;
  let sum2 = 0;
  const n = w * h;
  for (let i = 0; i < n; i++) {
    sum += gray[i];
    sum2 += gray[i] * gray[i];
  }
  contrastStd = Math.sqrt(Math.max(0, sum2 / n - (sum / n) ** 2));

  const blurScore = Math.min(1, lapVar / 400);
  const contrastScore = Math.min(1, contrastStd / 55);
  const squareScore = Math.min(1, Math.max(0, 1 - sqStd / 80));
  const gridScore = 0.55;
  const perspectiveScore = 0.5;
  const composite =
    0.28 * blurScore +
    0.18 * contrastScore +
    0.32 * gridScore +
    0.12 * squareScore +
    0.1 * perspectiveScore;

  return {
    warp_quality_score: Math.round(composite * 1000) / 10,
    blur_score: blurScore,
    contrast_score: contrastScore,
    grid_alignment_score: gridScore,
    square_consistency_score: squareScore,
    perspective_score: perspectiveScore,
    laplacian_variance: Math.round(lapVar * 10) / 10,
    contrast_std: Math.round(contrastStd * 10) / 10,
  };
}

function buildTrace(
  w: number,
  h: number,
  edgePts: CornerPoint[] | null,
  inset: CornerPoint[],
  selected: CornerPoint[],
  homography: number[][],
): GeometryTrace {
  const candidates: QuadCandidateTrace[] = [
    {
      points: inset,
      area: (w * h) * 0.76,
      aspect: 1,
      accepted: false,
      selected: false,
      rejectReason: "beaten_by_better_score",
      pass: "inset_fallback",
    },
  ];
  if (edgePts) {
    candidates.push({
      points: edgePts,
      area: w * h * 0.4,
      aspect: 1,
      accepted: true,
      selected: true,
      pass: "fast_edges",
    });
  } else {
    candidates[0] = { ...candidates[0], accepted: true, selected: true, rejectReason: undefined };
  }

  return {
    width: w,
    height: h,
    contours: edgePts
      ? [{ points: edgePts, area: w * h * 0.4 }]
      : [{ points: inset, area: w * h * 0.76 }],
    candidates,
    selectedContour: selected,
    rawCorners: edgePts ?? inset,
    orderedCorners: selected,
    homography,
  };
}

export function rectifyBoardFast(source: ImageData): FastRectifyResult {
  const { width: w, height: h, data } = source;
  const gray = grayscale(data, w, h);
  const blurred = boxBlur3(gray, w, h);
  const mag = sobelMag(blurred, w, h);

  const edgeCorners = cornersFromEdges(mag, w, h);
  const inset = insetCorners(w, h);
  const raw = edgeCorners ?? inset;
  const ordered = orderCorners(raw);

  const warpSize = Math.max(512, Math.min(640, Math.max(w, h)));
  const H = homographyFromQuad(ordered, warpSize);
  const Hinv = invert3x3(H);
  if (!Hinv) {
    throw new Error("Could not compute perspective transform");
  }

  const warpedRgba = warpPerspective(source, Hinv, warpSize);
  const warpedGray = grayscale(warpedRgba, warpSize, warpSize);
  const warpQuality = assessWarpQuality(warpedGray, warpSize, warpSize);
  const trace = buildTrace(w, h, edgeCorners, inset, ordered, H);

  return {
    corners: ordered,
    ordered,
    homography: H,
    warpedRgba,
    warpedWidth: warpSize,
    warpedHeight: warpSize,
    trace,
    warpQuality,
    source,
  };
}
