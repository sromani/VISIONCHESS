"use client";

import { useMemo } from "react";

import { SyntheticBoard } from "@/components/board/SyntheticBoard";
import { detectionsFromSquares } from "@/lib/chess/detections";
import type { DetectionResult } from "@/types";

/** Preview YOLO/classification result when FEN is not yet legal for interactive play. */
export function PendingPositionBoard({ detection }: { detection: DetectionResult }) {
  const detections = useMemo(
    () => detectionsFromSquares(detection.squares ?? []),
    [detection.squares],
  );

  const orientation =
    detection.orientation === "black" ? "black" : ("white" as const);

  return (
    <div className="flex w-full max-w-[min(100%,560px)] flex-col gap-2">
      <SyntheticBoard
        detections={detections}
        orientation={orientation}
        showControls
        showFen
        showHighlights
        highlightThreshold={0.45}
        boardId="PendingVisionBoard"
      />
      <p className="text-center text-xs text-muted">
        Low-confidence squares are highlighted. Use Setup to confirm or adjust.
      </p>
    </div>
  );
}
