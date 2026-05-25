import { useMemo, useRef, useEffect } from "react";

import { Button } from "@/components/ui/Button";
import type { LocalYoloResult } from "@/vision/yolo/types";

const BOX_COLORS = [
  "#22c55e",
  "#3b82f6",
  "#f59e0b",
  "#ef4444",
  "#a855f7",
  "#06b6d4",
  "#ec4899",
  "#84cc16",
];

export function LocalYoloDebugView({
  result,
  onBack,
}: {
  result: LocalYoloResult;
  onBack: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { metrics } = result;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = result.imageWidth;
      canvas.height = result.imageHeight;
      ctx.drawImage(img, 0, 0);
      result.detections.forEach((det, i) => {
        const { x, y, w, h } = det.bbox;
        const color = BOX_COLORS[i % BOX_COLORS.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, Math.round(Math.min(result.imageWidth, result.imageHeight) / 200));
        ctx.strokeRect(x, y, w, h);
        const label = `${det.className} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = `${Math.max(11, Math.round(result.imageWidth / 45))}px sans-serif`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - 18), tw + 8, 18);
        ctx.fillStyle = "#000";
        ctx.fillText(label, x + 4, Math.max(12, y - 4));
      });
    };
    img.src = result.imageUrl;
  }, [result]);

  const timingRows = useMemo(
    () => [
      ["Preprocess", `${metrics.preprocessMs.toFixed(1)} ms`],
      ["ONNX inference", `${metrics.inferenceMs.toFixed(1)} ms`],
      ["Postprocess", `${metrics.postprocessMs.toFixed(1)} ms`],
      ["Worker round-trip", `${metrics.workerRoundTripMs.toFixed(1)} ms`],
      ["Total (worker)", `${metrics.totalMs.toFixed(1)} ms`],
      ["Detections", String(metrics.detectionCount)],
      ["Image", `${result.imageWidth}×${result.imageHeight}`],
      ...(metrics.jsHeapUsedMb != null
        ? [["JS heap (approx)", `${metrics.jsHeapUsedMb} MB`]]
        : []),
    ],
    [metrics, result.imageWidth, result.imageHeight],
  );

  return (
    <div className="mx-auto flex w-full max-w-lg flex-col gap-4 py-4">
      <div className="text-center">
        <h2 className="text-xl font-semibold">Local YOLO (offline)</h2>
        <p className="mt-1 text-xs text-muted">
          Phase 1 — no FEN yet. Model: {result.modelPath}
        </p>
      </div>

      <div className="overflow-hidden rounded-2xl border border-border bg-black/40">
        <canvas ref={canvasRef} className="h-auto w-full" />
      </div>

      <div className="glass rounded-2xl p-4">
        <h3 className="text-sm font-semibold">Timing</h3>
        <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
          {timingRows.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted">{k}</dt>
              <dd className="font-mono text-foreground">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-[10px] text-muted">
          iOS no expone RAM ni temperatura al WebView. En Android/Chrome puede aparecer heap
          aproximado.
        </p>
      </div>

      <div className="glass max-h-48 overflow-y-auto rounded-2xl p-4">
        <h3 className="text-sm font-semibold">Detections</h3>
        {result.detections.length === 0 ? (
          <p className="mt-2 text-xs text-muted">No boxes above confidence threshold.</p>
        ) : (
          <ul className="mt-2 space-y-1 font-mono text-[11px]">
            {result.detections.map((d, i) => (
              <li key={`${d.bbox.x}-${d.bbox.y}-${i}`}>
                {d.className} {(d.confidence * 100).toFixed(1)}% — [{d.bbox.x}, {d.bbox.y},{" "}
                {d.bbox.w}×{d.bbox.h}]
              </li>
            ))}
          </ul>
        )}
      </div>

      <Button variant="secondary" className="touch-target w-full" onClick={onBack}>
        Back
      </Button>
    </div>
  );
}
