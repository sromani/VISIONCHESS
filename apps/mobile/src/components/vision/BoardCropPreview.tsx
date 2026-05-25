"use client";

import type { DetectionResult } from "@/types";

/** Shows full photo with crop overlay + rectified board when geometry succeeded. */
export function BoardCropPreview({ detection }: { detection: DetectionResult }) {
  const hasWarp =
    detection.corners.length === 4 &&
    detection.warpedUrl &&
    detection.warpedUrl !== detection.originalUrl;
  const method = detection.metadata?.geometryMethod as string | undefined;

  if (!hasWarp) return null;

  return (
    <div className="flex w-full max-w-[min(100%,560px)] flex-col gap-2">
      <p className="text-center text-xs text-muted">
        Tablero recortado de la foto
        {method === "yolo_board_bbox" ? " (detección YOLO del tablero)" : ""}
      </p>
      <div className="grid grid-cols-2 gap-2">
        <div className="overflow-hidden rounded-xl border border-border">
          <p className="bg-card/60 px-2 py-1 text-[10px] text-muted">Foto + marco</p>
          <img
            src={detection.overlayUrl}
            alt="Photo with board outline"
            className="h-auto w-full"
          />
        </div>
        <div className="overflow-hidden rounded-xl border border-border">
          <p className="bg-card/60 px-2 py-1 text-[10px] text-muted">Tablero rectificado</p>
          <img
            src={detection.warpedUrl}
            alt="Rectified board"
            className="h-auto w-full"
          />
        </div>
      </div>
    </div>
  );
}
