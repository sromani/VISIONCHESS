import { withTimeout } from "@/lib/async/withTimeout";
import type { BoardRegionResult } from "./yolo/boardRegion";
import { boardRegionFromDetections } from "./yolo/boardRegion";
import type { LocalYoloResult, YoloInferenceMetrics } from "./yolo/types";
import { YOLO_MODEL_PATH } from "./yolo/config";
import type { ImageRGBA } from "./yolo/types";

type Pending = {
  resolve: (value: LocalYoloResult) => void;
  reject: (reason: Error) => void;
  imageUrl: string;
  width: number;
  height: number;
  t0: number;
};

const INIT_TIMEOUT_MS = 15_000;
const INFERENCE_TIMEOUT_MS = 90_000;

let worker: Worker | null = null;
let readyPromise: Promise<void> | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function heapUsedMb(): number | undefined {
  const perf = performance as Performance & { memory?: { usedJSHeapSize: number } };
  if (!perf.memory) return undefined;
  return Math.round((perf.memory.usedJSHeapSize / (1024 * 1024)) * 10) / 10;
}

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./vision.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (ev: MessageEvent) => {
      const data = ev.data as {
        type: string;
        id?: number;
        detections?: LocalYoloResult["detections"];
        metrics?: Omit<
          YoloInferenceMetrics,
          "workerRoundTripMs" | "imageWidth" | "imageHeight" | "detectionCount" | "jsHeapUsedMb"
        >;
        message?: string;
      };

      if (data.type === "ready") {
        return;
      }

      if (data.type === "error") {
        const id = data.id;
        if (id != null && pending.has(id)) {
          const p = pending.get(id)!;
          pending.delete(id);
          p.reject(new Error(data.message ?? "Vision worker error"));
        }
        return;
      }

      if (data.type === "result" && data.id != null && pending.has(data.id)) {
        const p = pending.get(data.id)!;
        pending.delete(data.id);
        const workerRoundTripMs = performance.now() - p.t0;
        const detections = data.detections ?? [];
        const metrics: YoloInferenceMetrics = {
          ...data.metrics!,
          workerRoundTripMs,
          imageWidth: p.width,
          imageHeight: p.height,
          detectionCount: detections.length,
          jsHeapUsedMb: heapUsedMb(),
        };
        p.resolve({
          imageUrl: p.imageUrl,
          imageWidth: p.width,
          imageHeight: p.height,
          detections,
          metrics,
          modelPath: YOLO_MODEL_PATH,
        });
      }
    };
    worker.onerror = (e) => {
      const err = new Error(e.message || "Vision worker failed");
      for (const [, p] of pending) p.reject(err);
      pending.clear();
    };
  }
  return worker;
}

export async function initVisionWorker(): Promise<void> {
  if (readyPromise) {
    try {
      await readyPromise;
      return;
    } catch {
      readyPromise = null;
    }
  }

  readyPromise = new Promise((resolve, reject) => {
    const w = getWorker();
    const onReady = (ev: MessageEvent) => {
      if (ev.data?.type === "ready") {
        w.removeEventListener("message", onReady);
        resolve();
      } else if (ev.data?.type === "error") {
        w.removeEventListener("message", onReady);
        readyPromise = null;
        reject(new Error(ev.data.message ?? "Vision worker init failed"));
      }
    };
    w.addEventListener("message", onReady);
    w.postMessage({ type: "init" });
  });

  await withTimeout(readyPromise, INIT_TIMEOUT_MS, "Vision worker init");
}

export async function fileToImageData(file: File, maxSide = 1280): Promise<{
  imageUrl: string;
  width: number;
  height: number;
  data: Uint8ClampedArray;
}> {
  const imageUrl = URL.createObjectURL(file);
  const img = await loadImage(imageUrl);
  let { width, height } = img;
  const scale = maxSide / Math.max(width, height);
  if (scale < 1) {
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D not available");
  ctx.drawImage(img, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  return { imageUrl, width, height, data: imageData.data };
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to decode image"));
    img.src = url;
  });
}

async function runYoloMessage(
  image: ImageRGBA,
  previewUrl: string,
  mode: "pieces" | "board_scan",
): Promise<LocalYoloResult> {
  await initVisionWorker();
  const id = nextId++;
  const t0 = performance.now();
  const dataCopy = new Uint8ClampedArray(image.data);

  const inferencePromise = new Promise<LocalYoloResult>((resolve, reject) => {
    pending.set(id, {
      resolve: resolve as Pending["resolve"],
      reject,
      imageUrl: previewUrl,
      width: image.width,
      height: image.height,
      t0,
    });
    getWorker().postMessage(
      {
        type: "detect",
        id,
        mode,
        image: { width: image.width, height: image.height, data: dataCopy },
      },
      [dataCopy.buffer],
    );
  });
  return withTimeout(inferencePromise, INFERENCE_TIMEOUT_MS, "YOLO inference");
}

/** Find chessboard region in a full scene photo (YOLO class "board"). */
export async function detectBoardRegionInImage(
  image: ImageRGBA,
): Promise<BoardRegionResult | null> {
  const result = await runBoardScanRaw(image);
  return boardRegionFromDetections(result.detections, image.width, image.height);
}

/** Full board_scan output for WEB_DEBUG inspector. */
export async function runBoardScanRaw(image: ImageRGBA): Promise<LocalYoloResult> {
  return runYoloMessage(image, "", "board_scan");
}

export async function runYoloOnImage(
  image: { width: number; height: number; data: Uint8ClampedArray },
  previewUrl: string,
): Promise<LocalYoloResult> {
  return runYoloMessage(
    { width: image.width, height: image.height, data: image.data },
    previewUrl,
    "pieces",
  );
}

export async function runYoloInWorker(file: File): Promise<LocalYoloResult> {
  const { imageUrl, width, height, data } = await fileToImageData(file);
  return runYoloOnImage({ width, height, data }, imageUrl);
}

export function terminateVisionWorker(): void {
  worker?.terminate();
  worker = null;
  readyPromise = null;
  pending.clear();
}
