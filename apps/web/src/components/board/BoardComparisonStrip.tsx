"use client";

import { useMemo } from "react";

import { DetectionBoardGrid } from "@/components/board/DetectionBoardGrid";
import { SyntheticBoard } from "@/components/board/SyntheticBoard";
import {
  buildFen,
  detectionsFromSquares,
  lowConfidenceSquares,
  PieceDetection,
  BoardOrientation,
} from "@/lib/chess/detections";

function ComparisonFrame({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="flex flex-col overflow-hidden rounded-xl border border-border bg-card">
      <figcaption className="border-b border-border px-3 py-2">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted">{label}</span>
        {description && <p className="mt-0.5 text-[11px] text-muted/80">{description}</p>}
      </figcaption>
      <div className="flex flex-1 items-center justify-center bg-[#111] p-3">{children}</div>
    </figure>
  );
}

export function BoardComparisonStrip({
  originalUrl,
  rectifiedUrl,
  detections,
  fen,
  orientation = "white",
}: {
  originalUrl: string;
  rectifiedUrl: string;
  detections: PieceDetection[];
  fen?: string;
  orientation?: BoardOrientation;
}) {
  const highlightSquares = useMemo(
    () => new Set(lowConfidenceSquares(detections)),
    [detections],
  );
  const resolvedFen = fen ?? buildFen(detections, { orientation });

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-sm font-semibold tracking-tight">Board comparison</h2>
        <p className="text-xs text-muted">
          Photo → rectified warp → ML detections → synthetic render (SVG, no image crops)
        </p>
      </div>

      <div className="grid gap-3 xl:grid-cols-4 lg:grid-cols-2">
        <ComparisonFrame label="1. Original" description="Source photo for detection only">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={originalUrl} alt="Original photo" className="max-h-72 w-full object-contain" />
        </ComparisonFrame>

        <ComparisonFrame label="2. Rectified board" description="Perspective-corrected warp">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={rectifiedUrl} alt="Rectified board" className="max-h-72 w-full object-contain" />
        </ComparisonFrame>

        <ComparisonFrame label="3. Detections" description="Per-square ML labels + confidence">
          <DetectionBoardGrid detections={detections} highlightSquares={highlightSquares} />
        </ComparisonFrame>

        <ComparisonFrame label="4. Clean board" description="100% synthetic · lichess-style SVG">
          <div className="w-full max-w-[min(100%,320px)]">
            <SyntheticBoard
              detections={detections}
              fen={resolvedFen}
              orientation={orientation}
              showControls
              showFen={false}
              boardId="ComparisonSyntheticBoard"
            />
          </div>
        </ComparisonFrame>
      </div>

      <p className="font-mono text-[11px] text-muted">
        FEN placement: <span className="text-foreground">{resolvedFen.split(" ")[0]}</span>
      </p>
    </section>
  );
}

export { detectionsFromSquares };
