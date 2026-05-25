/**
 * Off-main-thread YOLO ONNX inference (onnxruntime-web).
 * Model loads on first detect — not during app bootstrap (100MB+ is too heavy on mobile).
 */
import * as ort from "onnxruntime-web/wasm";

import { configureOrtWasmEnv } from "./ortWasmEnv";
import { fetchArrayBuffer, workerAssetUrl } from "./workerAssets";
import { YOLO_CONFIG } from "./yolo/config";
import { postprocessYoloTensor } from "./yolo/postprocess";
import { preprocessYoloImage } from "./yolo/preprocess";
import type { ImageRGBA, YoloDetection, YoloInferenceMetrics } from "./yolo/types";

const YOLO_MODEL_URL = workerAssetUrl("models/yolov8_chess_pieces.onnx");

configureOrtWasmEnv(ort);

let session: ort.InferenceSession | null = null;
let sessionPromise: Promise<ort.InferenceSession> | null = null;

type WorkerIn =
  | { type: "init" }
  | {
      type: "detect";
      id: number;
      image: ImageRGBA;
      mode?: "pieces" | "board_scan";
    };

type WorkerOut =
  | { type: "ready" }
  | {
      type: "result";
      id: number;
      detections: YoloDetection[];
      metrics: Omit<
        YoloInferenceMetrics,
        "workerRoundTripMs" | "imageWidth" | "imageHeight" | "detectionCount"
      >;
    }
  | { type: "error"; id?: number; message: string };

async function ensureSession(): Promise<ort.InferenceSession> {
  if (session) return session;
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const buffer = await fetchArrayBuffer(YOLO_MODEL_URL, "YOLO model");
      return ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
      });
    })();
  }
  session = await sessionPromise;
  return session;
}

async function runDetect(
  image: ImageRGBA,
  mode: "pieces" | "board_scan" = "pieces",
): Promise<{
  detections: YoloDetection[];
  metrics: Omit<
    YoloInferenceMetrics,
    "workerRoundTripMs" | "imageWidth" | "imageHeight" | "detectionCount"
  >;
}> {
  const t0 = performance.now();
  const { blob, scale, dims } = preprocessYoloImage(image, YOLO_CONFIG.inputSize);
  const preprocessMs = performance.now() - t0;

  const sess = await ensureSession();
  const inputTensor = new ort.Tensor("float32", blob, dims);
  const t1 = performance.now();
  const outputs = await sess.run({ [YOLO_CONFIG.inputName]: inputTensor });
  const inferenceMs = performance.now() - t1;

  const outKey = Object.keys(outputs)[0];
  const tensor = outputs[outKey];
  const t2 = performance.now();
  const detections = postprocessYoloTensor(
    tensor.data as Float32Array,
    tensor.dims as number[],
    image.width,
    image.height,
    scale,
    mode === "board_scan"
      ? {
          skipClasses: new Set(),
          confThreshold: 0.2,
          maxBoxRatio: 0.92,
          minBoxPx: 8,
        }
      : undefined,
  );
  const postprocessMs = performance.now() - t2;
  const totalMs = performance.now() - t0;

  return {
    detections,
    metrics: {
      preprocessMs,
      inferenceMs,
      postprocessMs,
      totalMs,
    },
  };
}

self.onmessage = (ev: MessageEvent<WorkerIn>) => {
  const msg = ev.data;
  void (async () => {
    try {
      if (msg.type === "init") {
        self.postMessage({ type: "ready" } satisfies WorkerOut);
        return;
      }

      if (msg.type === "detect") {
        const { detections, metrics } = await runDetect(msg.image, msg.mode ?? "pieces");
        self.postMessage({
          type: "result",
          id: msg.id,
          detections,
          metrics,
        } satisfies WorkerOut);
        return;
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[Vision worker]", err);
      self.postMessage({
        type: "error",
        id: msg.type === "detect" ? msg.id : undefined,
        message,
      } satisfies WorkerOut);
    }
  })();
};
