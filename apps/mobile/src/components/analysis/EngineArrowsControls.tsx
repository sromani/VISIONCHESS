"use client";

import { cn, formatEval } from "@/lib/utils";
import { ENGINE_ARROW_COLORS } from "@/lib/chess/engineArrows";
import { useAppStore } from "@/store/appStore";
import type { EngineLine } from "@/types";

const MULTIPV_OPTIONS = [1, 2, 3] as const;

export function EngineArrowsControls({
  onHoverLine,
}: {
  onHoverLine?: (line: EngineLine | null) => void;
}) {
  const analysis = useAppStore((s) => s.analysis);
  const analysisLoading = useAppStore((s) => s.analysisLoading);
  const showEngineArrows = useAppStore((s) => s.showEngineArrows);
  const engineMultiPv = useAppStore((s) => s.engineMultiPv);
  const setShowEngineArrows = useAppStore((s) => s.setShowEngineArrows);
  const setEngineMultiPv = useAppStore((s) => s.setEngineMultiPv);

  const lines = analysis?.lines.slice(0, engineMultiPv) ?? [];

  return (
    <div className="space-y-3 rounded-xl border border-border bg-card/50 px-3 py-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={showEngineArrows}
            onChange={(e) => setShowEngineArrows(e.target.checked)}
            className="size-4 rounded border-border accent-accent"
          />
          <span>Show engine arrows</span>
        </label>

        <div className="flex items-center gap-1 rounded-lg border border-border bg-background/60 p-0.5">
          {MULTIPV_OPTIONS.map((count) => (
            <button
              key={count}
              type="button"
              disabled={!showEngineArrows}
              onClick={() => setEngineMultiPv(count)}
              className={cn(
                "rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
                engineMultiPv === count
                  ? "bg-accent text-accent-foreground"
                  : "text-muted hover:text-foreground",
                !showEngineArrows && "cursor-not-allowed opacity-40",
              )}
            >
              Top {count}
            </button>
          ))}
        </div>
      </div>

      {showEngineArrows && (
        <ul className="space-y-1.5">
          {lines.length === 0 && (
            <li className="text-center text-xs text-muted">
              {analysisLoading ? "Searching…" : "Engine lines will appear here"}
            </li>
          )}
          {lines.map((line, index) => (
            <li key={line.multipv}>
              <button
                type="button"
                title={`${line.san ?? line.move} (${formatEval(line.evalCp, line.evalMate)})`}
                onMouseEnter={() => onHoverLine?.(line)}
                onMouseLeave={() => onHoverLine?.(null)}
                onFocus={() => onHoverLine?.(line)}
                onBlur={() => onHoverLine?.(null)}
                className={cn(
                  "flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left",
                  "transition-colors hover:bg-accent/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent",
                )}
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span
                    className="size-2 shrink-0 rounded-full"
                    style={{ backgroundColor: ENGINE_ARROW_COLORS[index] }}
                  />
                  <span className="truncate font-mono text-sm text-foreground">
                    {line.san ?? line.move}
                  </span>
                </span>
                <span className="shrink-0 font-mono text-xs tabular-nums text-muted">
                  {formatEval(line.evalCp, line.evalMate)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
