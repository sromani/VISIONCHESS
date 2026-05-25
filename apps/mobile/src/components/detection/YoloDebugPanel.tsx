"use client";

import { DetectionResult } from "@/types";

function shortLabel(label: string): string {
  if (label === "empty" || label === "piece") return label;
  const parts = label.split("_");
  if (parts.length !== 2) return label;
  const abbr: Record<string, string> = {
    pawn: "P",
    knight: "N",
    bishop: "B",
    rook: "R",
    queen: "Q",
    king: "K",
  };
  return `${parts[0][0].toUpperCase()}${abbr[parts[1]] ?? parts[1][0].toUpperCase()}`;
}

export function YoloDebugPanel({ detection }: { detection: DetectionResult }) {
  const raw = detection.metadata?.yolo_detections;
  const detections = Array.isArray(raw)
    ? (raw as {
        label: string;
        confidence: number;
        square: string;
        bbox: number[];
        center?: number[];
      }[])
    : [];

  if (detections.length === 0 && !detection.debug?.mlPieceTop1) {
    return null;
  }

  return (
    <section className="animate-fade-up space-y-4 rounded-xl border border-border bg-card p-4">
      <div>
        <h2 className="text-sm font-semibold tracking-tight">YOLO piece detection</h2>
        <p className="text-xs text-muted">
          {String(detection.metadata?.piece_detector && typeof detection.metadata.piece_detector === "object"
            ? (detection.metadata.piece_detector as { source?: string }).source ?? "yolo"
            : "yolo")}{" "}
          · {detections.length} boxes · conf ≥{" "}
          {String(detection.metadata?.conf_threshold ?? 0.30)}
        </p>
      </div>

      {detection.debug?.mlPieceTop1 && (
        <figure className="overflow-hidden rounded-lg border border-border">
          <figcaption className="border-b border-border px-3 py-2 text-[11px] font-medium uppercase tracking-wider text-muted">
            Bounding boxes + mapped squares
          </figcaption>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={detection.debug.mlPieceTop1}
            alt="YOLO detections"
            className="w-full bg-[#111] object-contain"
          />
        </figure>
      )}

      {detections.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-border text-muted">
                <th className="px-2 py-2 font-medium">Square</th>
                <th className="px-2 py-2 font-medium">Class</th>
                <th className="px-2 py-2 font-medium">Conf</th>
                <th className="px-2 py-2 font-medium">BBox</th>
              </tr>
            </thead>
            <tbody>
              {detections.map((det) => (
                <tr key={`${det.square}-${det.label}-${det.confidence}`} className="border-b border-border/50">
                  <td className="px-2 py-1.5 font-mono">{det.square}</td>
                  <td className="px-2 py-1.5">{shortLabel(det.label)}</td>
                  <td className="px-2 py-1.5">{(det.confidence * 100).toFixed(1)}%</td>
                  <td className="px-2 py-1.5 font-mono text-muted">
                    [{det.bbox.join(", ")}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {detection.fen && (
        <p className="rounded-lg bg-background px-3 py-2 font-mono text-xs text-foreground">
          FEN: {detection.fen}
        </p>
      )}
    </section>
  );
}
