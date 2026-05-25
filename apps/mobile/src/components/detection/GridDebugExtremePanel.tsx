"use client";

import { cn } from "@/lib/utils";
import { DetectionDebug } from "@/types";

export function GridDebugExtremePanel({
  debug,
}: {
  debug: DetectionDebug | null | undefined;
}) {
  if (!debug?.gridDebugExtreme) return null;

  return (
    <section className={cn("animate-fade-up space-y-3")}>
      <div>
        <h2 className="text-sm font-semibold tracking-tight">Grid debug extremo</h2>
        <p className="text-xs text-muted">
          Líneas detectadas, polígonos exactos por casilla, centros e índices a1–h8
        </p>
      </div>
      <figure className="overflow-hidden rounded-xl border border-accent/30 bg-card">
        <div className="relative aspect-square bg-[#111]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={debug.gridDebugExtreme}
            alt="Extreme grid debug overlay"
            className="h-full w-full object-contain"
          />
        </div>
        <figcaption className="grid gap-1 border-t border-border px-3 py-2 text-[10px] text-muted sm:grid-cols-2">
          <span>■ Polígono blanco = casilla exacta</span>
          <span>■ Polígono cyan = crop interno</span>
          <span>+ Cruz roja = centro de casilla</span>
          <span>■ Amarillo = líneas X · Cyan = líneas Y</span>
        </figcaption>
      </figure>
    </section>
  );
}
