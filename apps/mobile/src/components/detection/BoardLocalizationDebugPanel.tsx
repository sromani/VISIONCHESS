"use client";

import type { DetectionResult } from "@/types";
import { detectionModeLabel } from "@/vision/detection/detectionMode";
import type { DetectionMode } from "@/vision/detection/detectionMode";

interface LocDebug {
  candidates: {
    source: string;
    score: number;
    lapsHits: number;
    lapsOk: boolean;
    geometry: number;
    grid: number;
    selected: boolean;
  }[];
  attempts: number;
  bestSource: string | null;
  rejectionReason?: string | null;
}

export function BoardLocalizationDebugPanel({ detection }: { detection: DetectionResult }) {
  const raw = detection.metadata?.board_localization;
  if (!raw || typeof raw !== "object") return null;

  const debug = raw as LocDebug;
  if (!debug.candidates?.length) return null;

  const score = detection.metadata?.localization_score;
  const attempts = detection.metadata?.localization_attempts;
  const mode = detection.metadata?.detection_mode as DetectionMode | undefined;
  const rejection =
    debug.rejectionReason ??
    (detection.metadata?.localization_rejection as string | null | undefined);

  return (
    <section className="animate-fade-up space-y-3 rounded-xl border border-border bg-card p-4">
      <div>
        <h2 className="text-sm font-semibold tracking-tight">Board localization</h2>
        <p className="text-xs text-muted">
          {mode ? `${mode} — ${detectionModeLabel(mode)}` : "Candidates"}
          {typeof score === "number" && ` · score ${Number(score).toFixed(3)}`}
          {typeof attempts === "number" && ` · ${attempts} attempt(s)`}
          {debug.bestSource && ` · ${debug.bestSource}`}
          {rejection && (
            <span className="block text-amber-400/90">Rejected / fail: {rejection}</span>
          )}
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[11px]">
          <thead>
            <tr className="border-b border-border text-muted">
              <th className="px-2 py-1.5">Source</th>
              <th className="px-2 py-1.5">Score</th>
              <th className="px-2 py-1.5">LAPS</th>
              <th className="px-2 py-1.5">Geom</th>
              <th className="px-2 py-1.5">Grid</th>
            </tr>
          </thead>
          <tbody>
            {debug.candidates.map((c) => (
              <tr
                key={`${c.source}-${c.score}`}
                className={
                  c.selected
                    ? "border-b border-emerald-500/30 bg-emerald-500/10"
                    : "border-b border-border/40 text-muted"
                }
              >
                <td className="px-2 py-1 font-mono">{c.source}</td>
                <td className="px-2 py-1">{c.score.toFixed(3)}</td>
                <td className="px-2 py-1">
                  {c.lapsHits}
                  {c.lapsOk ? " ✓" : ""}
                </td>
                <td className="px-2 py-1">{c.geometry.toFixed(2)}</td>
                <td className="px-2 py-1">{c.grid.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {detection.overlayUrl && (
        <figure className="overflow-hidden rounded-lg border border-border">
          <figcaption className="border-b border-border px-3 py-2 text-[11px] text-muted">
            Selected quad (green) + candidate scores
          </figcaption>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={detection.overlayUrl} alt="Board localization" className="w-full bg-[#111]" />
        </figure>
      )}
    </section>
  );
}
