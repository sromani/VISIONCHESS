export interface NativeYoloDetection {
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  square: string;
}

export interface OfflineRecognizeResult {
  placementFen: string;
  squares: Array<{
    name: string;
    label: string;
    confidence: number;
    occupied: boolean;
    bbox: [number, number, number, number];
  }>;
  detections: NativeYoloDetection[];
  timings: {
    preprocessMs: number;
    inferenceMs: number;
    postprocessMs: number;
    totalMs: number;
  };
  debug: {
    overlayJpegBase64: string;
    logLines: string[];
  };
}

export interface VisionChessOfflinePlugin {
  recognizeFromWarpedJpeg(options: {
    jpegBase64: string;
    width: number;
    height: number;
    confThreshold?: number;
  }): Promise<OfflineRecognizeResult>;
}
