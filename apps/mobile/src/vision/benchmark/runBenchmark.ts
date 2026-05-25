import {
  DETECTION_MODES,
  detectionModeLabel,
  getDetectionMode,
  type DetectionMode,
} from "@/vision/detection/detectionMode";
import { detectBoardFromFile } from "@/vision/detectBoardPipeline";

export interface BenchmarkManifest {
  version: number;
  description?: string;
  baselineMode: DetectionMode;
  images: string[];
}

export interface BenchmarkRow {
  image: string;
  mode: DetectionMode;
  success: boolean;
  boardFound: boolean;
  boardReady: boolean;
  localizationScore: number;
  pieceCount: number;
  geometryMs: number;
  totalMs: number;
  method: string;
  bestSource: string | null;
  rejection: string | null;
  error?: string;
}

export interface BenchmarkReport {
  runAt: string;
  baselineMode: DetectionMode;
  currentMode: DetectionMode;
  rows: BenchmarkRow[];
  summary: {
    total: number;
    success: number;
    boardFound: number;
    boardReady: number;
    avgMs: number;
  };
  vsBaseline?: {
    regressions: string[];
    improvements: string[];
  };
}

export async function loadBenchmarkManifest(): Promise<BenchmarkManifest> {
  const res = await fetch("/benchmark/manifest.json");
  if (!res.ok) {
    return { version: 1, baselineMode: "stable_v1", images: [] };
  }
  return res.json() as Promise<BenchmarkManifest>;
}

async function fetchImageFile(name: string): Promise<File> {
  const url = `/benchmark/images/${encodeURIComponent(name)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Missing ${url}`);
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type || "image/jpeg" });
}

export async function runBenchmarkForMode(
  mode: DetectionMode,
  images: string[],
  onProgress?: (done: number, total: number, image: string) => void,
): Promise<BenchmarkRow[]> {
  const rows: BenchmarkRow[] = [];
  for (let i = 0; i < images.length; i += 1) {
    const image = images[i];
    onProgress?.(i, images.length, image);
    const t0 = performance.now();
    try {
      const file = await fetchImageFile(image);
      const detection = await detectBoardFromFile({ file, mode });
      const totalMs = performance.now() - t0;
      rows.push({
        image,
        mode,
        success: true,
        boardFound: detection.metadata?.board_found === true,
        boardReady: detection.boardReady,
        localizationScore: Number(detection.metadata?.localization_score ?? 0),
        pieceCount: Number(detection.metadata?.yolo_piece_boxes ?? 0),
        geometryMs: Number(
          (detection.metadata?.geometryMs as number | undefined) ?? 0,
        ),
        totalMs: Math.round(totalMs),
        method: String(detection.metadata?.geometryMethod ?? ""),
        bestSource: (detection.metadata?.localization_best_source as string | null) ?? null,
        rejection: (detection.metadata?.localization_rejection as string | null) ?? null,
      });
    } catch (err) {
      rows.push({
        image,
        mode,
        success: false,
        boardFound: false,
        boardReady: false,
        localizationScore: 0,
        pieceCount: 0,
        geometryMs: 0,
        totalMs: Math.round(performance.now() - t0),
        method: "",
        bestSource: null,
        rejection: null,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }
  onProgress?.(images.length, images.length, "done");
  return rows;
}

function summarize(rows: BenchmarkRow[]): BenchmarkReport["summary"] {
  const ok = rows.filter((r) => r.success);
  return {
    total: rows.length,
    success: ok.length,
    boardFound: ok.filter((r) => r.boardFound).length,
    boardReady: ok.filter((r) => r.boardReady).length,
    avgMs: ok.length ? Math.round(ok.reduce((a, r) => a + r.totalMs, 0) / ok.length) : 0,
  };
}

function compareToBaseline(
  baseline: BenchmarkRow[],
  current: BenchmarkRow[],
): BenchmarkReport["vsBaseline"] {
  const regressions: string[] = [];
  const improvements: string[] = [];
  for (const b of baseline) {
    const c = current.find((r) => r.image === b.image);
    if (!c) continue;
    const bOk = b.boardFound || b.boardReady;
    const cOk = c.boardFound || c.boardReady;
    if (bOk && !cOk) regressions.push(b.image);
    if (!bOk && cOk) improvements.push(b.image);
  }
  return { regressions, improvements };
}

export async function runFullBenchmark(
  modes: DetectionMode[] = [...DETECTION_MODES],
): Promise<BenchmarkReport> {
  const manifest = await loadBenchmarkManifest();
  const images = manifest.images;
  const baselineMode = manifest.baselineMode ?? "stable_v1";
  const currentMode = getDetectionMode();

  const allRows: BenchmarkRow[] = [];
  let baselineRows: BenchmarkRow[] = [];

  for (const mode of modes) {
    const rows = await runBenchmarkForMode(mode, images);
    allRows.push(...rows);
    if (mode === baselineMode) baselineRows = rows;
  }

  const currentRows = allRows.filter((r) => r.mode === currentMode);

  return {
    runAt: new Date().toISOString(),
    baselineMode,
    currentMode,
    rows: allRows,
    summary: summarize(currentRows.length ? currentRows : allRows),
    vsBaseline:
      baselineRows.length && currentMode !== baselineMode
        ? compareToBaseline(baselineRows, currentRows)
        : undefined,
  };
}

export function downloadBenchmarkReport(report: BenchmarkReport): void {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `benchmark_${report.currentMode}_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export { DETECTION_MODES, detectionModeLabel, getDetectionMode };
