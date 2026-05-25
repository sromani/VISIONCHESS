"use client";

import { BoardComparisonStrip, detectionsFromSquares } from "@/components/board/BoardComparisonStrip";
import { cn } from "@/lib/utils";
import { PieceDetectionResult } from "@/types";

function shortLabel(label: string): string {
  if (label === "empty" || label === "piece") return label;
  const [color, piece] = label.split("_");
  const abbr: Record<string, string> = { pawn: "P", knight: "N", bishop: "B", rook: "R", queen: "Q", king: "K" };
  return `${color === "white" ? "W" : "B"}${abbr[piece] ?? piece[0].toUpperCase()}`;
}

export function PieceDetectionView({ result }: { result: PieceDetectionResult }) {
  const detections = detectionsFromSquares(
    result.squares.map((s) => ({
      squareName: s.squareName,
      label: s.label,
      confidence: s.confidence,
      occupied: s.occupied,
    })),
  );

  return (
    <div className="space-y-8">
      <BoardComparisonStrip
        originalUrl={result.originalUrl}
        rectifiedUrl={result.rectifiedBoardUrl}
        detections={detections}
      />

      <section className="space-y-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Two-stage detection</h2>
          <p className="text-xs text-muted">
            A. YOLO localizer ({result.detector.source}) · B. Classifier ({result.classifier.source}{" "}
            {result.classifier.image_size}px) · {result.occupiedCount} pieces · {result.processingMs}ms
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <figure className="overflow-hidden rounded-xl border border-border bg-[#111]">
            <figcaption className="border-b border-border px-3 py-2 text-[11px] font-medium">
              A. Localization — bbox + conf
            </figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.localizationOverlayUrl ?? result.rectifiedBoardUrl}
              alt="YOLO localization"
              className="w-full object-contain"
            />
          </figure>

          <figure className="overflow-hidden rounded-xl border border-border bg-[#111]">
            <figcaption className="border-b border-border px-3 py-2 text-[11px] font-medium">
              B. Classifier overlay — fine labels
            </figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.classifierOverlayUrl ?? result.rectifiedBoardUrl}
              alt="Classifier overlay"
              className="w-full object-contain"
            />
          </figure>

          {result.classifierCropsUrl && (
            <figure className="overflow-hidden rounded-xl border border-border bg-[#111] lg:col-span-1">
              <figcaption className="border-b border-border px-3 py-2 text-[11px] font-medium">
                C. Classifier crops montage
              </figcaption>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={result.classifierCropsUrl} alt="Classifier crops" className="w-full object-contain" />
            </figure>
          )}
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-tight">Per-piece: bbox → crop → top3</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {result.classifierCrops.map((item) => {
            const det = result.detections.find((d) => d.square === item.square);
            return (
              <div key={item.square} className="overflow-hidden rounded-xl border border-border bg-card">
                <div className="border-b border-border px-3 py-2 text-xs">
                  <span className="font-mono font-medium">{item.square}</span>
                  {det && (
                    <span className="ml-2 text-muted">
                      loc {det.localizationConfidence.toFixed(2)} · bbox [{det.bbox.join(",")}]
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2 p-2">
                  <div className="bg-[#111] p-1">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={item.cropUrl} alt={`Crop ${item.square}`} className="h-24 w-full object-contain" />
                    <p className="mt-1 text-center text-[9px] text-muted">crop exacto</p>
                  </div>
                  <div className="space-y-1 p-1 text-[10px]">
                    <p className="font-medium text-muted">top3 classifier</p>
                    {item.top3.map((t) => (
                      <div key={t.label} className="flex justify-between gap-2">
                        <span>{shortLabel(t.label)}</span>
                        <span>{(t.probability * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-tight">Square assignment grid</h2>
        <div className="grid grid-cols-8 gap-1 sm:max-w-2xl">
          {result.squares
            .slice()
            .sort((a, b) => a.row - b.row || a.col - b.col)
            .map((sq) => (
              <div
                key={sq.squareName}
                className={cn(
                  "rounded border border-border/60 px-1 py-1.5 text-center text-[9px] leading-tight",
                  sq.occupied ? "bg-emerald-950/40" : "bg-zinc-800/80 text-zinc-400",
                )}
              >
                <div className="font-mono text-[8px] text-muted">{sq.squareName}</div>
                <div className="font-semibold">{sq.occupied ? shortLabel(sq.label) : "empty"}</div>
                {sq.occupied && <div className="text-[8px]">{sq.confidence.toFixed(2)}</div>}
              </div>
            ))}
        </div>
      </section>
    </div>
  );
}
