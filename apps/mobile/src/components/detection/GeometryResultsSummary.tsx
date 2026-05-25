"use client";

import { useState } from "react";

import { OfflineGeometryDebugPanel } from "@/components/detection/OfflineGeometryDebugPanel";
import { cn } from "@/lib/utils";
import type { DetectionResult, WarpQualityMetrics } from "@/types";

function ResultImage({
  label,
  src,
  badge,
}: {
  label: string;
  src?: string;
  badge?: string;
}) {
  if (!src) return null;
  return (
    <figure className="min-w-0 flex-1 overflow-hidden rounded-lg border border-border bg-[#111]">
      <figcaption className="flex items-center justify-between gap-1 border-b border-border px-2 py-1.5">
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted">{label}</span>
        {badge && (
          <span className="rounded bg-violet-500/25 px-1.5 py-0.5 text-[9px] text-violet-200">
            {badge}
          </span>
        )}
      </figcaption>
      <img src={src} alt={label} className="aspect-square w-full object-contain" />
    </figure>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "ok" | "warn" | "muted" }) {
  return (
    <div className="rounded-lg border border-border bg-card/60 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-muted">{label}</p>
      <p
        className={cn(
          "mt-0.5 text-sm font-semibold tabular-nums",
          tone === "ok" && "text-emerald-400",
          tone === "warn" && "text-amber-400",
          (!tone || tone === "muted") && "text-foreground",
        )}
      >
        {value}
      </p>
    </div>
  );
}

export function GeometryResultsSummary({ detection }: { detection: DetectionResult }) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const meta = detection.metadata ?? {};
  const warp = meta.warp_quality as WarpQualityMetrics | undefined;
  const score =
    typeof warp?.warp_quality_score === "number"
      ? warp.warp_quality_score
      : typeof meta.warp_quality_score === "number"
        ? meta.warp_quality_score
        : null;

  const debug = detection.debug;
  const corners = detection.corners ?? [];
  const trace = meta.geometry_trace as { candidates?: { length: number } } | undefined;

  const original = debug?.original ?? detection.originalUrl;
  const rectified = debug?.rectifiedBoard ?? detection.warpedUrl;
  const grid = debug?.rectifiedGrid ?? detection.overlayUrl;

  return (
    <section className="w-full space-y-4 rounded-xl border border-emerald-500/30 bg-emerald-950/10 p-3">
      <div>
        <h3 className="text-base font-semibold text-emerald-100">Geometry results</h3>
        <p className="text-xs text-muted">
          What the fast pipeline produced from your photo (no pieces / FEN yet).
        </p>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-1">
        <ResultImage label="Original" src={original} />
        <ResultImage label="Rectified board" src={rectified} badge="warp" />
        <ResultImage label="8×8 grid" src={grid} badge="quality" />
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat
          label="Warp score"
          value={score !== null ? `${score.toFixed(0)} / 100` : "—"}
          tone={score !== null && score >= 60 ? "ok" : score !== null ? "warn" : "muted"}
        />
        <Stat label="Time" value={`${detection.processingMs ?? 0} ms`} />
        <Stat
          label="Backend"
          value={String(meta.geometry_backend ?? "?").replace(/_/g, " ")}
        />
        <Stat label="Candidates" value={String(trace?.candidates?.length ?? "—")} />
      </div>

      {corners.length >= 4 && (
        <div className="rounded-lg border border-border bg-black/30 px-3 py-2 text-[11px] text-muted">
          <p className="font-medium text-foreground">Ordered corners (TL → TR → BR → BL)</p>
          <ul className="mt-1 space-y-0.5 font-mono">
            {corners.slice(0, 4).map((c, i) => (
              <li key={i}>
                O{i}: ({Math.round(c.x)}, {Math.round(c.y)})
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex flex-wrap gap-2 text-[11px]">
        <span className={detection.boardReady ? "text-emerald-400" : "text-amber-400"}>
          board_ready={String(detection.boardReady)}
        </span>
        <span className={detection.fenValid ? "text-emerald-400" : "text-muted"}>
          fen_valid={String(detection.fenValid)}
        </span>
        <span className="text-muted">fen="{detection.fen || "(empty)"}"</span>
      </div>

      <button
        type="button"
        className="w-full rounded-lg border border-border bg-card/80 py-2 text-xs font-medium text-muted"
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced ? "Hide layer debug" : "Show layer debug (toggles & overlays)"}
      </button>

      {showAdvanced && <OfflineGeometryDebugPanel detection={detection} />}
    </section>
  );
}
