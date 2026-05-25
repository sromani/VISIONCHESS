export interface YoloBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface YoloDetection {
  label: string;
  className: string;
  confidence: number;
  bbox: YoloBBox;
}

export interface YoloInferenceMetrics {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  totalMs: number;
  workerRoundTripMs: number;
  imageWidth: number;
  imageHeight: number;
  detectionCount: number;
  /** Chrome/Android only; undefined on iOS Safari. */
  jsHeapUsedMb?: number;
}

export interface LocalYoloResult {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  detections: YoloDetection[];
  metrics: YoloInferenceMetrics;
  modelPath: string;
}

export interface ImageRGBA {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}
