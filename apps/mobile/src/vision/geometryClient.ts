import { withTimeout } from "@/lib/async/withTimeout";
import type { DetectionMode } from "./detection/detectionMode";
import type { ImageRGBA } from "./yolo/types";
import type { Point2, RectifyResult } from "./geometry/types";

const INIT_TIMEOUT_MS = 90_000;
const RECTIFY_TIMEOUT_MS = 120_000;

export type { Point2 };

type Pending = {
  resolve: (value: RectifyResult) => void;
  reject: (reason: Error) => void;
};

let worker: Worker | null = null;
let readyPromise: Promise<void> | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./geometry.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (ev: MessageEvent) => {
      const data = ev.data as { type: string; id?: number; result?: RectifyResult; message?: string };
      if (data.type === "ready") return;
      if (data.type === "error" && data.id != null && pending.has(data.id)) {
        const p = pending.get(data.id)!;
        pending.delete(data.id);
        p.reject(new Error(data.message ?? "Geometry worker error"));
        return;
      }
      if (data.type === "result" && data.id != null && pending.has(data.id)) {
        const p = pending.get(data.id)!;
        pending.delete(data.id);
        p.resolve(data.result!);
      }
    };
    worker.onerror = (e) => {
      const err = new Error(e.message || "Geometry worker failed");
      for (const [, p] of pending) p.reject(err);
      pending.clear();
    };
  }
  return worker;
}

export async function initGeometryWorker(): Promise<void> {
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
        reject(new Error(ev.data.message));
      }
    };
    w.addEventListener("message", onReady);
    w.postMessage({ type: "init" });
  });
  await withTimeout(readyPromise, INIT_TIMEOUT_MS, "Geometry worker init");
}

export async function rectifyBoardImage(
  image: ImageRGBA,
  options?: {
    corners?: Point2[];
    yoloConfidence?: number;
    detectionMode?: DetectionMode;
  },
): Promise<RectifyResult> {
  await initGeometryWorker();
  const id = nextId++;
  const rectifyPromise = new Promise<RectifyResult>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({
      type: "rectify",
      id,
      image,
      corners: options?.corners,
      yoloConfidence: options?.yoloConfidence,
      detectionMode: options?.detectionMode,
    });
  });
  return withTimeout(rectifyPromise, RECTIFY_TIMEOUT_MS, "Board localization");
}

export function imageDataToObjectUrl(image: ImageRGBA): string {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D not available");
  const pixels = new Uint8ClampedArray(image.data);
  ctx.putImageData(new ImageData(pixels, image.width, image.height), 0, 0);
  return canvas.toDataURL("image/jpeg", 0.92);
}

/** Corner overlay on original photo (LC2FEN-style). */
export function drawCornerOverlay(
  image: ImageRGBA,
  corners: Point2[],
): string {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D not available");
  const pixels = new Uint8ClampedArray(image.data);
  ctx.putImageData(new ImageData(pixels, image.width, image.height), 0, 0);
  if (corners.length === 4) {
    ctx.strokeStyle = "rgba(0, 220, 80, 0.95)";
    ctx.lineWidth = Math.max(2, Math.round(Math.min(image.width, image.height) / 200));
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i += 1) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.stroke();
    corners.forEach((c, idx) => {
      ctx.fillStyle = "rgba(0, 180, 255, 0.95)";
      ctx.beginPath();
      ctx.arc(c.x, c.y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 14px sans-serif";
      ctx.fillText(String(idx), c.x + 10, c.y - 6);
    });
  }
  return canvas.toDataURL("image/jpeg", 0.92);
}

/** Debug: candidate scores list + selected board quad (green). */
export function drawLocalizationDebugOverlay(
  image: ImageRGBA,
  debug: import("./geometry/types").LocalizationDebug,
  winnerCorners: Point2[],
): string {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D not available");
  const pixels = new Uint8ClampedArray(image.data);
  ctx.putImageData(new ImageData(pixels, image.width, image.height), 0, 0);

  ctx.font = `${Math.max(10, Math.round(image.width / 90))}px sans-serif`;
  let ty = 16;
  for (const cand of debug.candidates.slice(0, 10)) {
    const label = `${cand.selected ? "★" : "·"} ${cand.source} s=${cand.score} L=${cand.lapsHits} g=${cand.grid}`;
    ctx.fillStyle = cand.selected ? "rgba(0,220,80,0.95)" : "rgba(255,255,255,0.8)";
    ctx.fillText(label, 8, ty);
    ty += 14;
  }

  if (winnerCorners.length === 4) {
    ctx.strokeStyle = "rgba(0, 220, 80, 0.95)";
    ctx.lineWidth = Math.max(2, Math.round(Math.min(image.width, image.height) / 200));
    ctx.beginPath();
    ctx.moveTo(winnerCorners[0].x, winnerCorners[0].y);
    for (let i = 1; i < 4; i += 1) ctx.lineTo(winnerCorners[i].x, winnerCorners[i].y);
    ctx.closePath();
    ctx.stroke();
  }
  return canvas.toDataURL("image/jpeg", 0.92);
}
