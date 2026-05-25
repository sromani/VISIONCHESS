"use client";

import { useMemo, useState } from "react";

import { cn } from "@/lib/utils";
import type { DetectionResult, WarpQualityMetrics } from "@/types";

export function RectifiedResultBoard({ detection }: { detection: DetectionResult }) {
  const [showGrid, setShowGrid] = useState(true);

  const meta = detection.metadata ?? {};
  const warp = meta.warp_quality as WarpQualityMetrics | undefined;
  const score =
    typeof warp?.warp_quality_score === "number"
      ? warp.warp_quality_score
      : typeof meta.warp_quality_score === "number"
        ? meta.warp_quality_score
        : null;

  const plainSrc = detection.debug?.rectifiedBoard ?? detection.warpedUrl;
  const gridSrc = detection.debug?.rectifiedGrid ?? detection.overlayUrl ?? plainSrc;
  const displaySrc = showGrid && gridSrc ? gridSrc : plainSrc;

  const scoreTone = useMemo(() => {
    if (score === null) return "muted" as const;
    if (score >= 60) return "ok" as const;
    return "warn" as const;
  }, [score]);

  if (!plainSrc) {
    return (
      <div className="w-full max-w-[min(100%,560px)] rounded-2xl border border-dashed border-amber-500/40 bg-card/40 px-4 py-10 text-center">
        <p className="text-sm font-medium text-foreground">No rectified board image</p>
        <p className="mt-2 text-xs text-muted">Warp may have failed — check metrics below.</p>
      </div>
    );
  }

  return (
    <div className="flex w-full max-w-[min(100%,560px)] flex-col gap-2">
      <div className="flex items-center justify-between gap-2 px-0.5">
        <div>
          <h3 className="text-sm font-semibold tracking-tight text-foreground">Result board</h3>
          <p className="text-[11px] text-muted">Top-down warp from your photo</p>
        </div>
        {score !== null && (
          <span
            className={cn(
              "rounded-full px-2.5 py-1 text-xs font-semibold tabular-nums",
              scoreTone === "ok" && "bg-emerald-500/20 text-emerald-300",
              scoreTone === "warn" && "bg-amber-500/20 text-amber-300",
              scoreTone === "muted" && "bg-muted/20 text-muted",
            )}
          >
            {score.toFixed(0)}%
          </span>
        )}
      </div>

      <div
        className={cn(
          "overflow-hidden rounded-2xl border border-border shadow-[0_8px_40px_rgb(0_0_0/0.12)]",
          "dark:shadow-[0_8px_40px_rgb(0_0_0/0.45)]",
        )}
      >
        <div className="aspect-square w-full bg-[#1a1a1a]">
          <img
            src={displaySrc}
            alt={showGrid ? "Rectified board with 8×8 grid" : "Rectified board"}
            className="h-full w-full object-contain"
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        <button
          type="button"
          className={cn(
            "rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
            !showGrid
              ? "border-violet-500/50 bg-violet-500/15 text-violet-100"
              : "border-border bg-card/80 text-muted",
          )}
          onClick={() => setShowGrid(false)}
        >
          Warp only
        </button>
        <button
          type="button"
          className={cn(
            "rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
            showGrid
              ? "border-violet-500/50 bg-violet-500/15 text-violet-100"
              : "border-border bg-card/80 text-muted",
          )}
          onClick={() => setShowGrid(true)}
        >
          With 8×8 grid
        </button>
      </div>
    </div>
  );
}
