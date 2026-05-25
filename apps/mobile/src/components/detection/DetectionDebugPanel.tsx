"use client";

import { cn } from "@/lib/utils";
import { DetectionDebug } from "@/types";

const DEBUG_STEPS: { key: keyof DetectionDebug; label: string; description: string }[] = [
  { key: "original", label: "1. Original", description: "Source photo" },
  { key: "detectedLines", label: "2. Detected lines", description: "9×9 H/V grid lines" },
  { key: "intersections", label: "3. Intersections", description: "81 mesh points" },
  { key: "mesh", label: "4. Mesh", description: "Source playing mesh" },
  { key: "rectifiedBoard", label: "5. Rectified board", description: "Canonical mesh preview" },
  { key: "rectifiedUpscaled", label: "6. Upscaled board", description: "2048px super-resolution" },
  { key: "squareExtraction", label: "7. Raw crops", description: "High-res square splits" },
  { key: "cropQuality", label: "8. Crop quality", description: "Enhanced ML input crops" },
  { key: "occupancy", label: "9. Occupancy", description: "Soft probability heatmap" },
  { key: "occupancyDetail", label: "10. Occupancy signals", description: "Per-square fusion breakdown" },
  { key: "classifierConfidence", label: "11. Classifier", description: "Piece confidence heatmap" },
  { key: "mlOccupancy", label: "12. ML occupancy", description: "Raw ONNX P(occupied) per square" },
  { key: "mlPieceTop1", label: "13. ML piece top-1", description: "Best piece class + confidence" },
  { key: "mlOnnxOccCrops", label: "14. Occ ONNX crops", description: "100×100 inputs to occupancy model" },
  { key: "mlOnnxPieceCrops", label: "15. Piece ONNX crops", description: "299×299 inputs to piece model" },
  { key: "mlDetail", label: "16. ML logits", description: "Top squares with logits + top-3" },
  { key: "fenCandidates", label: "17. FEN hypotheses", description: "Legality + Stockfish ranking" },
  { key: "finalBoard", label: "18. Final board", description: "Validated classification grid" },
];

function DebugFrame({ label, description, src }: { label: string; description: string; src: string }) {
  return (
    <figure className="overflow-hidden rounded-xl border border-border bg-card">
      <figcaption className="border-b border-border px-3 py-2">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted">{label}</span>
        <p className="mt-0.5 text-[11px] text-muted/80">{description}</p>
      </figcaption>
      <div className="relative aspect-[4/3] bg-[#111]">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={label} className="h-full w-full object-contain" />
      </div>
    </figure>
  );
}

export function DetectionDebugPanel({ debug }: { debug: DetectionDebug | null | undefined }) {
  if (!debug) return null;

  const available = DEBUG_STEPS.filter((step) => debug[step.key]);

  if (available.length === 0) return null;

  return (
    <section className={cn("animate-fade-up space-y-3")}>
      <div>
        <h2 className="text-sm font-semibold tracking-tight">Scanner pipeline debug</h2>
        <p className="text-xs text-muted">
          Geometry → high-res crops → ML occupancy/classify → hypotheses → validated FEN
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {available.map((step) => (
          <DebugFrame
            key={step.key}
            label={step.label}
            description={step.description}
            src={debug[step.key]!}
          />
        ))}
      </div>
    </section>
  );
}
