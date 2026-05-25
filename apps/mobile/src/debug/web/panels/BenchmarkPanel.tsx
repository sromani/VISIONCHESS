"use client";

import { useState } from "react";

import { Button } from "@/components/ui/Button";
import {
  clearFailureCases,
  downloadFailureExport,
  listFailureCases,
} from "@/vision/debug/failureCapture";
import {
  detectionModeLabel,
  DETECTION_MODES,
  loadBenchmarkManifest,
  runBenchmarkForMode,
  type BenchmarkRow,
} from "@/vision/benchmark/runBenchmark";
import { useWebDebugStore } from "@/debug/web/webDebugStore";

export function BenchmarkPanel() {
  const detectionMode = useWebDebugStore((s) => s.detectionMode);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [rows, setRows] = useState<BenchmarkRow[]>([]);
  const [compareMode, setCompareMode] = useState<string>("stable_v1");
  const [failCount, setFailCount] = useState(listFailureCases().length);

  async function run(mode: string) {
    setRunning(true);
    setRunError(null);
    setProgress("Loading manifest…");
    try {
      const manifest = await loadBenchmarkManifest();
      if (!manifest.images.length) {
        setRunError(
          "No hay imágenes en manifest.json. Agregá archivos en public/benchmark/images/.",
        );
        return;
      }
      setProgress(`0 / ${manifest.images.length} — first run can take 1–2 min (YOLO load)`);
      const result = await runBenchmarkForMode(
        mode as (typeof DETECTION_MODES)[number],
        manifest.images,
        (done, total, image) => {
          setProgress(`${done + 1} / ${total}${image !== "done" ? ` — ${image}` : ""}`);
        },
      );
      setRows(result);
      setProgress(null);
    } catch (err) {
      setRunError(err instanceof Error ? err.message : String(err));
      setProgress(null);
    } finally {
      setRunning(false);
      setFailCount(listFailureCases().length);
    }
  }

  return (
    <section className="space-y-3 rounded-lg border border-border/80 bg-card/50 p-3">
      <div>
        <h2 className="text-xs font-semibold tracking-tight">Benchmark</h2>
        <p className="mt-0.5 text-[10px] text-muted">
          Mode: {detectionMode} — {detectionModeLabel(detectionMode)}
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <select
          className="rounded border border-border bg-background px-2 py-1 text-[10px]"
          value={compareMode}
          onChange={(e) => setCompareMode(e.target.value)}
        >
          {DETECTION_MODES.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <Button size="sm" disabled={running} onClick={() => void run(compareMode)}>
          {running ? "…" : `Run ${compareMode}`}
        </Button>
      </div>

      {progress && <p className="text-[10px] text-muted">{progress}</p>}
      {runError && <p className="text-[10px] text-red-400">{runError}</p>}

      {rows.length > 0 && (
        <div className="max-h-48 overflow-auto text-[10px]">
          {rows.map((r) => (
            <div key={`${r.image}-${r.mode}`} className="border-b border-border/30 py-0.5">
              {r.image} board={r.boardFound ? "Y" : "N"} {r.totalMs}ms
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2">
        <Button size="sm" variant="ghost" onClick={downloadFailureExport}>
          Export fails
        </Button>
        <Button size="sm" variant="ghost" onClick={() => { clearFailureCases(); setFailCount(0); }}>
          Clear ({failCount})
        </Button>
      </div>
    </section>
  );
}
