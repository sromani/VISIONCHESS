"use client";

import type { DetectionResult } from "@/types";
import { detectionModeLabel, DETECTION_MODES } from "@/vision/detection/detectionMode";
import type { DetectionMode } from "@/vision/detection/detectionMode";

import { BenchmarkPanel } from "./panels/BenchmarkPanel";
import { useWebDebugStore } from "./webDebugStore";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-border/60 py-3">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted">
        {title}
      </h3>
      {children}
    </section>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="mb-2 block text-[11px]">
      <span className="flex justify-between text-muted">
        <span>{label}</span>
        <span className="font-mono text-foreground">{value.toFixed(2)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full accent-emerald-500"
      />
    </label>
  );
}

interface LocCandidate {
  source: string;
  score: number;
  lapsHits: number;
  lapsOk: boolean;
  geometry: number;
  grid: number;
  selected: boolean;
}

export function WebDebugSidebar({
  detection,
  phase,
  pipelineSteps,
}: {
  detection: DetectionResult | null;
  phase: string;
  pipelineSteps: { id: string; label: string; status: string }[];
}) {
  const { detectionMode, thresholds, setDetectionMode, setThreshold, resetThresholds } =
    useWebDebugStore();

  const loc = detection?.metadata?.board_localization as
    | { candidates: LocCandidate[]; bestSource: string | null; rejectionReason?: string | null }
    | undefined;

  const yoloRaw = detection?.metadata?.yolo_detections as
    | { label: string; confidence: number; square: string; bbox: number[] }[]
    | undefined;

  const boardScan = detection?.metadata?.yolo_board_scan as
    | { label: string; confidence: number; bbox: number[] }[]
    | undefined;

  const rejected = loc?.candidates?.filter((c) => !c.selected) ?? [];
  const metrics = detection?.metadata?.yolo_metrics as
    | { preprocessMs?: number; inferenceMs?: number; postprocessMs?: number; totalMs?: number }
    | undefined;

  return (
    <aside className="flex h-full w-[min(100%,380px)] shrink-0 flex-col overflow-hidden border-r border-border bg-[#0c0c0e] lg:w-[400px]">
      <header className="shrink-0 border-b border-border px-4 py-3">
        <p className="text-sm font-semibold text-foreground">CV Lab</p>
        <p className="text-[10px] text-muted">WEB_DEBUG — not shipped on mobile</p>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4">
        <Section title="1. Original image">
          {detection?.originalUrl ? (
            <img
              src={detection.originalUrl}
              alt="Original"
              className="w-full rounded-lg border border-border bg-black/40"
            />
          ) : (
            <p className="text-xs text-muted">Upload an image to inspect.</p>
          )}
        </Section>

        <Section title="2–4. Board candidates & selection">
          {loc?.candidates?.length ? (
            <>
              <table className="w-full text-left text-[10px]">
                <thead>
                  <tr className="text-muted">
                    <th className="py-1">Src</th>
                    <th>Score</th>
                    <th>LAPS</th>
                  </tr>
                </thead>
                <tbody>
                  {loc.candidates.map((c) => (
                    <tr
                      key={`${c.source}-${c.score}`}
                      className={c.selected ? "text-emerald-400" : "text-muted"}
                    >
                      <td className="py-0.5 font-mono">{c.source}</td>
                      <td>{c.score.toFixed(3)}</td>
                      <td>
                        {c.lapsHits}
                        {c.lapsOk ? " ✓" : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-[10px] text-muted">
                Selected: {loc.bestSource ?? "—"}
                {loc.rejectionReason && (
                  <span className="block text-amber-400">Reject: {loc.rejectionReason}</span>
                )}
              </p>
              {detection?.overlayUrl && (
                <img
                  src={detection.overlayUrl}
                  alt="Selected board quad"
                  className="mt-2 w-full rounded border border-emerald-500/30"
                />
              )}
            </>
          ) : (
            <p className="text-xs text-muted">No localization debug yet.</p>
          )}
        </Section>

        <Section title="5–7. Bounding boxes & YOLO">
          {detection?.debug?.mlPieceTop1 && (
            <img
              src={detection.debug.mlPieceTop1}
              alt="Piece bboxes"
              className="mb-2 w-full rounded border border-border"
            />
          )}
          {boardScan && boardScan.length > 0 && (
            <div className="mb-2">
              <p className="mb-1 text-[10px] font-medium text-muted">Board scan (raw)</p>
              <ul className="max-h-24 overflow-y-auto font-mono text-[10px] text-foreground">
                {boardScan.map((d, i) => (
                  <li key={i}>
                    {d.label} {(d.confidence * 100).toFixed(0)}% bbox=
                    {d.bbox.map((n) => Math.round(n)).join(",")}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {yoloRaw && yoloRaw.length > 0 ? (
            <ul className="max-h-32 overflow-y-auto font-mono text-[10px]">
              {yoloRaw.map((d) => (
                <li key={`${d.square}-${d.label}`} className="text-foreground">
                  {d.square} {d.label} {(d.confidence * 100).toFixed(0)}%
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-muted">No piece detections.</p>
          )}
          {rejected.length > 0 && (
            <p className="mt-2 text-[10px] text-amber-400/90">
              Rejected candidates: {rejected.map((c) => c.source).join(", ")}
            </p>
          )}
        </Section>

        <Section title="8. Threshold sliders">
          <label className="mb-2 block text-[11px] text-muted">
            Detection mode
            <select
              className="mt-1 w-full rounded border border-border bg-background px-2 py-1 text-xs text-foreground"
              value={detectionMode}
              onChange={(e) => setDetectionMode(e.target.value as DetectionMode)}
            >
              {DETECTION_MODES.map((m) => (
                <option key={m} value={m}>
                  {m} — {detectionModeLabel(m)}
                </option>
              ))}
            </select>
          </label>
          <p className="mb-2 text-[10px] text-muted">Re-upload image after changing thresholds.</p>
          <SliderRow
            label="YOLO conf"
            value={thresholds.yoloConf}
            min={0.05}
            max={0.9}
            step={0.05}
            onChange={(v) => setThreshold("yoloConf", v)}
          />
          <SliderRow
            label="YOLO IoU"
            value={thresholds.yoloIou}
            min={0.1}
            max={0.9}
            step={0.05}
            onChange={(v) => setThreshold("yoloIou", v)}
          />
          <SliderRow
            label="Board conf"
            value={thresholds.boardConf}
            min={0.05}
            max={0.9}
            step={0.05}
            onChange={(v) => setThreshold("boardConf", v)}
          />
          <SliderRow
            label="Min board area %"
            value={thresholds.minBoardAreaRatio}
            min={0.01}
            max={0.5}
            step={0.01}
            onChange={(v) => setThreshold("minBoardAreaRatio", v)}
          />
          <button
            type="button"
            className="text-[10px] text-muted underline"
            onClick={resetThresholds}
          >
            Reset thresholds
          </button>
        </Section>

        <Section title="9. Inference timings (ms)">
          <ul className="font-mono text-[11px] text-foreground">
            <li>Geometry: {String(detection?.metadata?.geometryMs ?? "—")}</li>
            <li>YOLO preprocess: {metrics?.preprocessMs?.toFixed(1) ?? "—"}</li>
            <li>YOLO inference: {metrics?.inferenceMs?.toFixed(1) ?? "—"}</li>
            <li>YOLO postprocess: {metrics?.postprocessMs?.toFixed(1) ?? "—"}</li>
            <li>YOLO total: {metrics?.totalMs?.toFixed(1) ?? "—"}</li>
            <li>Pipeline: {detection?.processingMs ?? "—"}</li>
          </ul>
        </Section>

        <Section title="10. Pipeline stages">
          <p className="mb-1 font-mono text-[11px] text-emerald-400">phase: {phase}</p>
          <ul className="space-y-1 text-[11px]">
            {pipelineSteps.map((s) => (
              <li
                key={s.id}
                className={
                  s.status === "active"
                    ? "text-emerald-400"
                    : s.status === "done"
                      ? "text-muted"
                      : "text-muted/50"
                }
              >
                {s.label} — {s.status}
              </li>
            ))}
          </ul>
          {detection && (
            <pre className="mt-2 max-h-40 overflow-auto rounded bg-black/40 p-2 text-[9px] text-muted">
              {JSON.stringify(
                {
                  mode: detection.metadata?.detection_mode,
                  board_found: detection.metadata?.board_found,
                  method: detection.metadata?.geometryMethod,
                  score: detection.metadata?.localization_score,
                  boardReady: detection.boardReady,
                },
                null,
                2,
              )}
            </pre>
          )}
        </Section>

        <div className="py-3">
          <BenchmarkPanel />
        </div>
      </div>
    </aside>
  );
}
