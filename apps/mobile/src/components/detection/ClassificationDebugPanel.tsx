"use client";

import { DetectionDebug } from "@/types";
import { DetectionBoardGrid } from "@/components/board/DetectionBoardGrid";
import { SyntheticBoard } from "@/components/board/SyntheticBoard";
import {
  buildFen,
  detectionsFromSquares,
  lowConfidenceSquares,
} from "@/lib/chess/detections";

export function ClassificationDebugPanel({
  squares,
  orientation,
  fenValid,
  fen,
}: {
  debug?: DetectionDebug | null;
  squares?: { name: string; label?: string; confidence?: number; occupied?: boolean }[];
  orientation?: string;
  fenValid?: boolean;
  fen?: string;
}) {
  if (squares?.length) {
    const detections = detectionsFromSquares(
      squares.map((sq) => ({
        name: sq.name,
        label: sq.label ?? "empty",
        confidence: sq.confidence ?? 0,
        occupied: sq.occupied,
      })),
    );
    const boardOrientation = orientation === "flipped" ? "black" : "white";
    const resolvedFen = fen ?? buildFen(detections, { orientation: boardOrientation });
    const highlightSquares = new Set(lowConfidenceSquares(detections));

    return (
      <section className="animate-fade-up space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold tracking-tight">Synthetic board from detections</h2>
            <p className="text-xs text-muted">
              SVG render · no photo crops
              {orientation && ` · orientation: ${orientation}`}
              {fenValid !== undefined && (fenValid ? " · FEN valid" : " · FEN needs review")}
            </p>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <figure className="overflow-hidden rounded-xl border border-border bg-card p-4">
            <figcaption className="mb-3 text-[11px] font-medium text-muted">Detection grid</figcaption>
            <DetectionBoardGrid detections={detections} highlightSquares={highlightSquares} />
          </figure>

          <figure className="overflow-hidden rounded-xl border border-border bg-card p-4">
            <figcaption className="mb-3 text-[11px] font-medium text-muted">Clean board</figcaption>
            <SyntheticBoard
              detections={detections}
              fen={resolvedFen}
              orientation={boardOrientation}
              boardId="ClassificationSyntheticBoard"
            />
          </figure>
        </div>

        <p className="font-mono text-[11px] text-muted">{resolvedFen.split(" ")[0]}</p>
      </section>
    );
  }

  return null;
}
