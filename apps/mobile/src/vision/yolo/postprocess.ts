import { YOLO_CONFIG, YOLO_CLASS_NAMES } from "./config";
import type { YoloDetection } from "./types";

export interface PostprocessOptions {
  confThreshold: number;
  iouThreshold: number;
  inputSize: number;
  classNames: readonly string[];
  skipClasses: Set<string>;
  localizationOnly: boolean;
  boardClassIndex: number;
  coordsNormalized: boolean;
  maxBoxRatio: number;
  minBoxPx: number;
}

const DEFAULT_OPTS: PostprocessOptions = {
  confThreshold: YOLO_CONFIG.confThreshold,
  iouThreshold: YOLO_CONFIG.iouThreshold,
  inputSize: YOLO_CONFIG.inputSize,
  classNames: YOLO_CLASS_NAMES,
  skipClasses: YOLO_CONFIG.skipClasses,
  localizationOnly: YOLO_CONFIG.localizationOnly,
  boardClassIndex: YOLO_CONFIG.boardClassIndex,
  coordsNormalized: YOLO_CONFIG.coordsNormalized,
  maxBoxRatio: YOLO_CONFIG.maxBoxRatio,
  minBoxPx: YOLO_CONFIG.minBoxPx,
};

/** Port of `_postprocess` + helpers in apps/api/vision/inference/yolo_detector.py */
export function postprocessYoloOutput(
  output: Float32Array | number[],
  origW: number,
  origH: number,
  scale: number,
  options: Partial<PostprocessOptions> = {},
): YoloDetection[] {
  const opts = { ...DEFAULT_OPTS, ...options };
  const pred = reshapePredictions(output, opts.inputSize);
  if (!pred || pred.numRows < 1 || pred.numCols < 5) {
    return [];
  }

  const boxesXywh: number[][] = [];
  const classScores: number[][] = [];
  for (let i = 0; i < pred.numRows; i += 1) {
    boxesXywh.push(Array.from(pred.row(i).slice(0, 4)));
    classScores.push(Array.from(pred.row(i).slice(4)));
  }

  let normalized = opts.coordsNormalized;
  let maxCoord = 0;
  for (const b of boxesXywh) {
    for (const v of b) maxCoord = Math.max(maxCoord, v);
  }
  if (maxCoord <= 1.5) normalized = true;

  const scaleCoords = (boxes: number[][]) =>
    normalized ? boxes.map((b) => b.map((v) => v * opts.inputSize)) : boxes;

  const scaledBoxes = scaleCoords(boxesXywh);

  const confidences: number[] = [];
  const classIds: number[] = [];
  const labels: string[] = [];

  for (let i = 0; i < pred.numRows; i += 1) {
    const scores = classScores[i];
    let conf: number;
    let cid: number;

    if (
      opts.localizationOnly &&
      scores.length > 1 &&
      opts.boardClassIndex >= 0
    ) {
      const pieceScores = scores.filter((_, j) => j !== opts.boardClassIndex);
      conf = Math.max(...pieceScores);
      cid = pieceScores.indexOf(conf);
      if (cid >= opts.boardClassIndex) cid += 1;
    } else {
      cid = 0;
      for (let j = 1; j < scores.length; j += 1) {
        if (scores[j] > scores[cid]) cid = j;
      }
      conf = scores[cid];
    }

    const className =
      cid < opts.classNames.length ? opts.classNames[cid] : `class_${cid}`;
    confidences.push(conf);
    classIds.push(cid);
    labels.push(className);
  }

  const filtered: number[] = [];
  for (let i = 0; i < pred.numRows; i += 1) {
    if (!opts.skipClasses.has(labels[i]) && confidences[i] >= opts.confThreshold) {
      filtered.push(i);
    }
  }

  if (filtered.length === 0) return [];

  let boxesXyxy = filtered.map((i) => xywhToXyxy(scaledBoxes[i]));
  const filtConf = filtered.map((i) => confidences[i]);
  const filtLabels = filtered.map((i) => labels[i]);

  boxesXyxy = boxesXyxy.map((b) => [
    clip(b[0] / scale, 0, origW),
    clip(b[1] / scale, 0, origH),
    clip(b[2] / scale, 0, origW),
    clip(b[3] / scale, 0, origH),
  ]);

  const maxW = origW * opts.maxBoxRatio;
  const maxH = origH * opts.maxBoxRatio;
  const sizeKept: number[] = [];
  for (let i = 0; i < boxesXyxy.length; i += 1) {
    const [x1, y1, x2, y2] = boxesXyxy[i];
    const bw = x2 - x1;
    const bh = y2 - y1;
    if (bw >= opts.minBoxPx && bh >= opts.minBoxPx && bw <= maxW && bh <= maxH) {
      sizeKept.push(i);
    }
  }

  if (sizeKept.length === 0) return [];

  const finalBoxes = sizeKept.map((i) => boxesXyxy[i]);
  const finalConf = sizeKept.map((i) => filtConf[i]);
  const finalLabels = sizeKept.map((i) => filtLabels[i]);

  const keep = nms(finalBoxes, finalConf, opts.iouThreshold);
  const detections: YoloDetection[] = [];

  for (const idx of keep) {
    const className = finalLabels[idx];
    const displayLabel = opts.localizationOnly ? "piece" : className;
    const [x1, y1, x2, y2] = finalBoxes[idx];
    detections.push({
      label: displayLabel,
      className,
      confidence: finalConf[idx],
      bbox: {
        x: Math.round(x1),
        y: Math.round(y1),
        w: Math.max(1, Math.round(x2 - x1)),
        h: Math.max(1, Math.round(y2 - y1)),
      },
    });
  }

  detections.sort((a, b) => b.confidence - a.confidence);
  return detections;
}

class PredictionMatrix {
  constructor(
    public readonly numRows: number,
    public readonly numCols: number,
    private readonly flat: Float32Array,
  ) {}

  row(i: number): Float32Array {
    const start = i * this.numCols;
    return this.flat.subarray(start, start + this.numCols);
  }
}

function reshapePredictions(
  output: Float32Array | number[],
  inputSize: number,
): PredictionMatrix | null {
  const data = output instanceof Float32Array ? output : Float32Array.from(output);

  // ONNX [1, 4+C, N] or [1, N, 4+C]
  if (data.length % (4 + 13) === 0 || data.length > inputSize) {
    // Heuristic: try common YOLOv8 layout [1, 17, 8400]
    const n17 = 17;
    if (data.length % n17 === 0) {
      const anchors = data.length / n17;
      if (anchors > n17) {
        return new PredictionMatrix(anchors, n17, data);
      }
      return new PredictionMatrix(n17, anchors, transposeFlat(data, n17, anchors));
    }
  }

  return null;
}

function transposeFlat(data: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      out[c * rows + r] = data[r * cols + c];
    }
  }
  return out;
}

/** Match Python `_reshape_predictions` on 3D batch output. */
export function reshapeYoloOnnxOutput(output: Float32Array, dims: readonly number[]): PredictionMatrix | null {
  if (dims.length < 2) return null;

  let rows = 0;
  let cols = 0;
  let matrix = output;

  if (dims.length === 3) {
    const d1 = dims[1];
    const d2 = dims[2];
    if (d1 <= 64 && d2 > d1) {
      rows = d2;
      cols = d1;
      matrix = transposeFlat(output, d1, d2);
    } else if (d2 <= 64 && d1 > d2) {
      rows = d1;
      cols = d2;
    } else {
      rows = d1;
      cols = d2;
    }
  } else if (dims.length === 2) {
    rows = dims[0];
    cols = dims[1];
    if (rows <= 64 && cols > rows) {
      matrix = transposeFlat(output, rows, cols);
      rows = cols;
      cols = dims[0];
    } else if (cols <= 64 && rows > cols) {
      matrix = transposeFlat(output, rows, cols);
      rows = dims[1];
      cols = dims[0];
    }
  }

  if (rows < 1 || cols < 5) return null;
  return new PredictionMatrix(rows, cols, matrix);
}

export function postprocessYoloTensor(
  output: Float32Array,
  dims: readonly number[],
  origW: number,
  origH: number,
  scale: number,
  options: Partial<PostprocessOptions> = {},
): YoloDetection[] {
  const pred = reshapeYoloOnnxOutput(output, dims);
  if (!pred) return [];
  const flat: number[] = [];
  for (let i = 0; i < pred.numRows; i += 1) {
    flat.push(...pred.row(i));
  }
  return postprocessYoloOutput(flat, origW, origH, scale, options);
}

function xywhToXyxy(box: number[]): [number, number, number, number] {
  const [cx, cy, w, h] = box;
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

function nms(boxes: number[][], scores: number[], iouThreshold: number): number[] {
  const order = scores
    .map((_, i) => i)
    .sort((a, b) => scores[b] - scores[a]);
  const keep: number[] = [];

  while (order.length > 0) {
    const i = order[0];
    keep.push(i);
    if (order.length === 1) break;

    const remaining = order.slice(1);
    const next: number[] = [];
    const [x1, y1, x2, y2] = boxes[i];
    const areaI = (x2 - x1) * (y2 - y1);

    for (const j of remaining) {
      const [xx1, yy1, xx2, yy2] = boxes[j];
      const interW = Math.max(0, Math.min(x2, xx2) - Math.max(x1, xx1));
      const interH = Math.max(0, Math.min(y2, yy2) - Math.max(y1, yy1));
      const inter = interW * interH;
      const areaJ = (xx2 - xx1) * (yy2 - yy1);
      const iou = inter / (areaI + areaJ - inter + 1e-6);
      if (iou <= iouThreshold) next.push(j);
    }
    order.length = 0;
    order.push(...next);
  }

  return keep;
}

function clip(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
