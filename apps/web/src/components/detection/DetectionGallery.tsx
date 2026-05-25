"use client";

import { ClassificationDebugPanel } from "@/components/detection/ClassificationDebugPanel";
import { DetectionDebugPanel } from "@/components/detection/DetectionDebugPanel";
import { GridDebugExtremePanel } from "@/components/detection/GridDebugExtremePanel";
import { MlDebugPanel } from "@/components/detection/MlDebugPanel";
import { YoloDebugPanel } from "@/components/detection/YoloDebugPanel";
import { BoardComparisonStrip, detectionsFromSquares } from "@/components/board/BoardComparisonStrip";
import { cn } from "@/lib/utils";
import { DetectionResult } from "@/types";

function ImageFrame({
  label,
  src,
  badge,
  className,
}: {
  label: string;
  src: string;
  badge?: string;
  className?: string;
}) {
  return (
    <figure className={cn("overflow-hidden rounded-xl border border-border bg-card", className)}>
      <figcaption className="flex items-center justify-between border-b border-border px-3 py-2">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted">{label}</span>
        {badge && (
          <span className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-medium text-accent">
            {badge}
          </span>
        )}
      </figcaption>
      <div className="relative aspect-square bg-[#111]">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={label} className="h-full w-full object-contain" />
      </div>
    </figure>
  );
}

export function DetectionGallery({ detection }: { detection: DetectionResult }) {
  const detections = detectionsFromSquares(
    detection.squares.map((sq) => ({
      name: sq.name,
      label: sq.label ?? "empty",
      confidence: sq.confidence ?? 0,
      occupied: sq.occupied,
    })),
  );
  const boardOrientation = detection.orientation === "flipped" ? "black" : "white";

  return (
    <div className="space-y-8">
      <BoardComparisonStrip
        originalUrl={detection.originalUrl}
        rectifiedUrl={detection.warpedUrl}
        detections={detections}
        fen={detection.interactiveFen ?? detection.fen}
        orientation={boardOrientation}
      />

      <section className="animate-fade-up space-y-3">
        <div className="flex items-end justify-between">
          <div>
            <h2 className="text-sm font-semibold tracking-tight">Results</h2>
            <p className="text-xs text-muted">
              Confidence {(detection.confidence * 100).toFixed(0)}% · {detection.processingMs}ms
            </p>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <ImageFrame label="Original" src={detection.originalUrl} />
          <ImageFrame label="Top-down warp" src={detection.warpedUrl} badge="Final" />
          <ImageFrame label="Grid overlay" src={detection.overlayUrl} badge="Post-warp" />
        </div>
      </section>

      <YoloDebugPanel detection={detection} />

      <GridDebugExtremePanel debug={detection.debug} />

      <DetectionDebugPanel debug={detection.debug} />

      <MlDebugPanel mlDebug={detection.mlDebug} squares={detection.squares} debug={detection.debug} />

      <ClassificationDebugPanel
        debug={detection.debug}
        squares={detection.squares}
        orientation={detection.orientation}
        fenValid={detection.fenValid}
        fen={detection.interactiveFen ?? detection.fen}
      />
    </div>
  );
}
