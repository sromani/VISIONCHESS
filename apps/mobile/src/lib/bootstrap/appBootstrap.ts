import { withTimeout } from "@/lib/async/withTimeout";
import { VISION_LOCAL } from "@/lib/config";
import { prepareStockfishEngine } from "@/lib/chess/stockfishEngine";
import { initGeometryWorker } from "@/vision/geometryClient";
import { initVisionWorker } from "@/vision/visionClient";

export const BOOTSTRAP_MIN_MS = 1500;
export const BOOTSTRAP_FADE_MS = 420;

/** Worker spawn only — YOLO ONNX (~100MB) loads on first scan, not here. */
const VISION_WORKER_TIMEOUT_MS = 15_000;
const GEOMETRY_INIT_TIMEOUT_MS = 90_000;
const STOCKFISH_INIT_TIMEOUT_MS = 20_000;
export const BOOTSTRAP_HARD_CAP_MS = 10_000;

export interface BootstrapProgress {
  id: string;
  label: string;
  done: boolean;
}

export interface BootstrapResult {
  steps: BootstrapProgress[];
  warnings: string[];
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/**
 * Preload offline runtime before showing the main UI.
 */
export async function runAppBootstrap(
  onStep: (steps: BootstrapProgress[]) => void,
): Promise<BootstrapResult> {
  const started = performance.now();
  const steps: BootstrapProgress[] = [
    { id: "stockfish", label: "Stockfish engine", done: false },
  ];

  if (VISION_LOCAL) {
    steps.push(
      { id: "vision", label: "Vision worker ready", done: false },
      { id: "geometry", label: "Board geometry worker", done: false },
    );
  }

  steps.push({ id: "runtime", label: "Offline runtime", done: false });
  onStep([...steps]);

  const mark = (id: string) => {
    const s = steps.find((x) => x.id === id);
    if (s) s.done = true;
    onStep([...steps]);
  };

  const warnings: string[] = [];

  try {
    await withTimeout(
      prepareStockfishEngine(),
      STOCKFISH_INIT_TIMEOUT_MS,
      "Stockfish",
    );
  } catch (err) {
    warnings.push(err instanceof Error ? err.message : String(err));
    console.warn("[bootstrap] Stockfish", err);
  }
  mark("stockfish");

  if (VISION_LOCAL) {
    try {
      await withTimeout(initVisionWorker(), VISION_WORKER_TIMEOUT_MS, "Vision worker");
    } catch (err) {
      warnings.push(err instanceof Error ? err.message : String(err));
      console.warn("[bootstrap] Vision worker", err);
    }
    mark("vision");

    mark("geometry");
    void withTimeout(initGeometryWorker(), GEOMETRY_INIT_TIMEOUT_MS, "Geometry worker").catch(
      (err) => {
        const msg = err instanceof Error ? err.message : String(err);
        warnings.push(msg);
        console.warn("[bootstrap] Geometry worker (background)", err);
      },
    );
  }

  mark("runtime");

  const elapsed = performance.now() - started;
  if (elapsed < BOOTSTRAP_MIN_MS) {
    await sleep(BOOTSTRAP_MIN_MS - elapsed);
  }

  return { steps, warnings };
}
