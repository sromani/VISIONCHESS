"use client";

import { useMemo } from "react";

import { SyntheticBoard } from "@/components/board/SyntheticBoard";
import {
  detectionHasPieces,
  detectionsFromDetectionResult,
  type BoardOrientation,
} from "@/lib/chess/detections";
import { cn } from "@/lib/utils";
import type { DetectionResult } from "@/types";

export function DetectedPositionBoard({
  detection,
  className,
}: {
  detection: DetectionResult;
  className?: string;
}) {
  const hasPieces = detectionHasPieces(detection);
  const detections = useMemo(() => detectionsFromDetectionResult(detection), [detection]);
  const orientation: BoardOrientation =
    detection.orientation === "flipped" ? "black" : "white";
  const fen = detection.interactiveFen ?? detection.fen ?? undefined;

  if (!hasPieces) {
    return (
      <div
        className={cn(
          "w-full max-w-[min(100%,560px)] rounded-2xl border border-dashed border-amber-500/35 bg-amber-950/15 px-4 py-6 text-center",
          className,
        )}
      >
        <p className="text-sm font-medium text-foreground">No pieces detected yet</p>
        <p className="mt-2 text-xs text-muted">
          Set <code className="text-[10px]">VITE_API_URL</code> in{" "}
          <code className="text-[10px]">apps/mobile/.env.development</code> and restart dev
          server so YOLO can classify squares after the warp.
        </p>
      </div>
    );
  }

  return (
    <div className={cn("flex w-full max-w-[min(100%,560px)] flex-col gap-2", className)}>
      <div className="px-0.5">
        <h3 className="text-sm font-semibold tracking-tight text-foreground">Detected position</h3>
        <p className="text-[11px] text-muted">
          Clean board drawn from ML labels
          {detection.boardReady ? " · valid for play" : " · preview (FEN not validated)"}
        </p>
      </div>
      <SyntheticBoard
        detections={detections.length ? detections : undefined}
        fen={fen}
        orientation={orientation}
        showControls
        showFen
        showHighlights
        boardId="DetectedPositionBoard"
      />
    </div>
  );
}
