"use client";

import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { MlSquareDebug, SquareInfo } from "@/types";

function shortLabel(label: string): string {
  if (label === "empty") return "empty";
  const [color, piece] = label.split("_");
  const abbr: Record<string, string> = {
    pawn: "P",
    knight: "N",
    bishop: "B",
    rook: "R",
    queen: "Q",
    king: "K",
  };
  return `${color === "white" ? "w" : "b"}${abbr[piece] ?? piece[0]}`;
}

function SquareChip({
  name,
  selected,
  occupiedProb,
  isOccupied,
  topLabel,
  topProb,
  onClick,
}: {
  name: string;
  selected: boolean;
  occupiedProb?: number;
  isOccupied?: boolean;
  topLabel?: string;
  topProb?: number;
  onClick: () => void;
}) {
  const heat = occupiedProb ?? 0;
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-center justify-center rounded border px-0.5 py-1 text-[9px] leading-tight transition",
        selected ? "border-accent bg-accent/20 ring-1 ring-accent" : "border-border/60 hover:border-accent/50",
        isOccupied === false ? "bg-zinc-800/90 text-zinc-400" : "bg-card/80",
      )}
      style={
        selected || isOccupied === false
          ? undefined
          : {
              backgroundColor: `rgba(${Math.round(heat * 180)}, ${Math.round((1 - heat) * 120)}, 40, 0.35)`,
            }
      }
    >
      <span className="font-mono text-[8px] text-muted">{name}</span>
      <span className="font-medium">{isOccupied ? (topLabel ? shortLabel(topLabel) : "—") : "empty"}</span>
      <span className="text-muted">{isOccupied && topProb !== undefined ? topProb.toFixed(2) : "—"}</span>
      <span className="text-[7px] text-muted">occ:{heat.toFixed(2)}</span>
    </button>
  );
}

export function MlDebugPanel({
  mlDebug,
  squares,
  debug,
  squareFusion,
  fusedEmptyThreshold = 0.35,
}: {
  mlDebug: Record<string, MlSquareDebug> | null | undefined;
  squares: SquareInfo[];
  debug: {
    mlOnnxOccCrops?: string;
    mlOnnxPieceCrops?: string;
    mlDetail?: string;
    mlPieceTop1?: string;
    mlOccupancy?: string;
  } | null | undefined;
  squareFusion?: Record<
    string,
    {
      occupancyProbability: number;
      pieceLabel: string;
      pieceConfidence: number;
      fusedConfidence: number;
      label: string;
      occupied: boolean;
    }
  >;
  fusedEmptyThreshold?: number;
}) {
  const [selected, setSelected] = useState("e4");

  const squareNames = useMemo(() => Object.keys(mlDebug ?? {}).sort(), [mlDebug]);
  const active = mlDebug?.[selected];

  if (!mlDebug || squareNames.length === 0) return null;

  return (
    <section className="animate-fade-up space-y-4">
      <div>
        <h2 className="text-sm font-semibold tracking-tight">ML debug — chesscog ONNX</h2>
        <p className="text-xs text-muted">
          Soft fusion · empty &lt; {fusedEmptyThreshold.toFixed(2)} · piece ONNX on all squares · occupancy = prior
        </p>
      </div>

      <div className="grid grid-cols-8 gap-1 sm:max-w-md">
        {squareNames.map((name) => {
          const sq = mlDebug[name];
          const fusion = squareFusion?.[name];
          const occP = fusion?.occupancyProbability ?? sq.occupancy?.occupied_probability ?? 0;
          const isOccupied = fusion?.occupied ?? occP > fusedEmptyThreshold;
          const topLabel = fusion?.pieceLabel ?? sq.piece?.top3?.[0]?.label;
          const topProb = fusion?.pieceConfidence ?? sq.piece?.top3?.[0]?.probability;
          return (
            <SquareChip
              key={name}
              name={name}
              selected={name === selected}
              occupiedProb={occP}
              isOccupied={isOccupied}
              topLabel={topLabel}
              topProb={topProb}
              onClick={() => setSelected(name)}
            />
          );
        })}
      </div>

      {active && (
        <div className="grid gap-4 rounded-xl border border-border bg-card p-4 lg:grid-cols-2">
          <div className="space-y-3">
            <h3 className="font-mono text-sm font-semibold">{selected}</h3>
            {active.occupancy && (
              <div className="space-y-1 text-xs">
                <p className="font-medium text-sky-300">Occupancy prior (ResNet {active.occupancy.model_input_size}px)</p>
                <p>
                  P(occupied) = <strong>{active.occupancy.occupied_probability.toFixed(4)}</strong>
                </p>
                <p className="font-mono text-muted">
                  logits = [{active.occupancy.logits.map((v) => v.toFixed(3)).join(", ")}]
                </p>
              </div>
            )}
            {active.piece && (
              <div className="space-y-1 text-xs">
                <p className="font-medium text-emerald-300">
                  Piece top-1 (InceptionV3 {active.piece.model_input_size}px)
                </p>
                <ol className="list-decimal space-y-1 pl-4">
                  {active.piece.top3.map((p) => (
                    <li key={p.label}>
                      <strong>{p.label}</strong> — prob {p.probability.toFixed(4)}, logit {p.logit.toFixed(3)}
                    </li>
                  ))}
                </ol>
              </div>
            )}
            {squareFusion?.[selected] && (
              <div className="space-y-1 rounded border border-border/60 p-2 text-xs">
                <p className="font-medium text-amber-200">Fused decision</p>
                <p>
                  fused = {squareFusion[selected].fusedConfidence.toFixed(4)} →{" "}
                  <strong>{squareFusion[selected].label}</strong>
                </p>
              </div>
            )}
            <p className="text-[10px] text-muted">
              analysis crop: {active.analysis_crop_shape[0]}×{active.analysis_crop_shape[1]} px
            </p>
          </div>

          <div className="grid grid-cols-3 gap-2">
            {[
              { title: "Analysis crop", src: squares.find((s) => s.name === selected) },
              { title: "Occ ONNX input", src: null },
              { title: "Piece ONNX input", src: null },
            ].map(({ title }) => (
              <figure key={title} className="overflow-hidden rounded-lg border border-border bg-[#111]">
                <figcaption className="border-b border-border px-2 py-1 text-[10px] text-muted">{title}</figcaption>
                <div className="flex aspect-square items-center justify-center p-1">
                  {title === "Analysis crop" ? (
                    <span className="text-[10px] text-muted">via square split</span>
                  ) : (
                    <span className="text-[10px] text-muted">see montage ↓</span>
                  )}
                </div>
              </figure>
            ))}
          </div>
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2">
        {debug?.mlOnnxOccCrops && (
          <figure className="overflow-hidden rounded-xl border border-border">
            <figcaption className="border-b border-border px-3 py-2 text-[11px] font-medium">ONNX occupancy inputs (100×100)</figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={debug.mlOnnxOccCrops} alt="ONNX occupancy crops" className="w-full bg-[#111] object-contain" />
          </figure>
        )}
        {debug?.mlOnnxPieceCrops && (
          <figure className="overflow-hidden rounded-xl border border-border">
            <figcaption className="border-b border-border px-3 py-2 text-[11px] font-medium">ONNX piece inputs (299×299)</figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={debug.mlOnnxPieceCrops} alt="ONNX piece crops" className="w-full bg-[#111] object-contain" />
          </figure>
        )}
        {debug?.mlDetail && (
          <figure className="overflow-hidden rounded-xl border border-border sm:col-span-2">
            <figcaption className="border-b border-border px-3 py-2 text-[11px] font-medium">Top squares — logits detail</figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={debug.mlDetail} alt="ML detail panel" className="w-full bg-[#111] object-contain" />
          </figure>
        )}
      </div>
    </section>
  );
}
