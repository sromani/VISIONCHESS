import {
  mergeMultiPvToAnalysisResult,
  parseBestMoveLine,
  parseUciInfoLine,
  type ParsedUciInfo,
  uciToSan,
} from "@/lib/chess/uciParse";
import type { AnalysisResult } from "@/types";

const DEFAULT_DEPTH = 12;
const INIT_TIMEOUT_MS = 15000;

export interface AnalyzeOptions {
  depth?: number;
  multiPv?: number;
  onUpdate?: (analysis: AnalysisResult) => void;
  signal?: AbortSignal;
}

function wasmSupported(): boolean {
  return (
    typeof WebAssembly === "object" &&
    typeof WebAssembly.validate === "function" &&
    WebAssembly.validate(Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0))
  );
}

function workerUrl(): string {
  return wasmSupported() ? "/stockfish.wasm.js" : "/stockfish.js";
}

export class StockfishEngine {
  private worker: Worker | null = null;
  private initPromise: Promise<void> | null = null;
  private searchGeneration = 0;
  private activeGeneration = 0;
  private configuredMultiPv = 1;

  /** Warm up UCI worker (call during app bootstrap). */
  prepare(): Promise<void> {
    return this.ensureReady();
  }

  private ensureReady(): Promise<void> {
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise((resolve, reject) => {
      try {
        this.worker = new Worker(workerUrl());
      } catch (error) {
        reject(error);
        return;
      }

      let sawUciOk = false;
      const timeout = window.setTimeout(() => {
        reject(new Error("Stockfish init timeout"));
      }, INIT_TIMEOUT_MS);

      const onInitMessage = (line: string) => {
        if (line.includes("uciok")) sawUciOk = true;
        if (sawUciOk && line.includes("readyok")) {
          window.clearTimeout(timeout);
          this.worker!.removeEventListener("message", initListener);
          resolve();
        }
      };

      const initListener = (event: MessageEvent<string>) => {
        onInitMessage(event.data);
      };

      this.worker.addEventListener("message", initListener);
      this.worker.onerror = (event) => {
        window.clearTimeout(timeout);
        reject(new Error(event.message || "Stockfish worker error"));
      };

      this.worker.postMessage("uci");
      this.worker.postMessage("isready");
    });

    return this.initPromise;
  }

  private configureMultiPv(multiPv: number): void {
    if (!this.worker || multiPv === this.configuredMultiPv) return;
    this.configuredMultiPv = multiPv;
    this.worker.postMessage(`setoption name MultiPV value ${multiPv}`);
  }

  stop(): void {
    this.searchGeneration += 1;
    this.worker?.postMessage("stop");
  }

  async analyze(fen: string, options: AnalyzeOptions = {}): Promise<AnalysisResult | null> {
    await this.ensureReady();
    if (!this.worker || options.signal?.aborted) return null;

    const depth = options.depth ?? DEFAULT_DEPTH;
    const multiPv = Math.max(1, Math.min(3, options.multiPv ?? 1));
    const generation = ++this.searchGeneration;
    this.activeGeneration = generation;
    const startedAt = Date.now();

    this.worker.postMessage("stop");
    this.configureMultiPv(multiPv);
    this.worker.postMessage(`position fen ${fen}`);
    this.worker.postMessage(`go depth ${depth}`);

    return new Promise((resolve) => {
      let latest: AnalysisResult | null = null;
      const linesByPv = new Map<number, ParsedUciInfo>();

      const cleanup = () => {
        this.worker?.removeEventListener("message", onSearchMessage);
        options.signal?.removeEventListener("abort", onAbort);
      };

      const finish = (result: AnalysisResult | null) => {
        if (generation !== this.activeGeneration) return;
        cleanup();
        resolve(result);
      };

      const onAbort = () => {
        this.stop();
        finish(latest);
      };

      options.signal?.addEventListener("abort", onAbort, { once: true });

      const publish = () => {
        if (linesByPv.size === 0) return;
        latest = mergeMultiPvToAnalysisResult(fen, linesByPv, startedAt);
        options.onUpdate?.(latest);
      };

      const onSearchMessage = (event: MessageEvent<string>) => {
        if (generation !== this.activeGeneration) return;

        const line = event.data;
        const bestMove = parseBestMoveLine(line);
        if (bestMove) {
          if (latest) {
            latest = {
              ...latest,
              bestMove,
              bestMoveSan: uciToSan(fen, bestMove),
              processingMs: Date.now() - startedAt,
            };
          }
          finish(latest);
          return;
        }

        const info = parseUciInfoLine(line);
        if (!info) return;
        if (info.evaluationCp === null && info.evaluationMate === null && !info.bestMoveUci) {
          return;
        }

        const previous = linesByPv.get(info.multipv);
        linesByPv.set(info.multipv, {
          ...previous,
          ...info,
          bestMoveUci: info.bestMoveUci || previous?.bestMoveUci || "",
          pv: info.pv.length > 0 ? info.pv : (previous?.pv ?? []),
          nodes: info.nodes ?? previous?.nodes ?? null,
          nps: info.nps ?? previous?.nps ?? null,
        });

        publish();
      };

      this.worker!.addEventListener("message", onSearchMessage);
    });
  }

  dispose(): void {
    this.stop();
    this.worker?.postMessage("quit");
    this.worker?.terminate();
    this.worker = null;
    this.initPromise = null;
  }
}

let engineSingleton: StockfishEngine | null = null;

export function getStockfishEngine(): StockfishEngine {
  if (!engineSingleton) {
    engineSingleton = new StockfishEngine();
  }
  return engineSingleton;
}

export function prepareStockfishEngine(): Promise<void> {
  return getStockfishEngine().prepare();
}

export async function analyzePosition(
  fen: string,
  options: AnalyzeOptions = {},
): Promise<AnalysisResult | null> {
  return getStockfishEngine().analyze(fen, options);
}
