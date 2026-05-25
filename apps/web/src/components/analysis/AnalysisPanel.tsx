"use client";

import { cn, formatEval, formatNodes, evalToPercent } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";
import { AnalysisResult } from "@/types";

export function EvalBarVertical({
  analysis,
  loading,
}: {
  analysis: AnalysisResult | null;
  loading?: boolean;
}) {
  const cp = analysis?.evaluationCp ?? null;
  const mate = analysis?.evaluationMate ?? null;
  const whitePct = evalToPercent(cp, mate);

  return (
    <div
      className="relative w-5 shrink-0 overflow-hidden rounded-lg bg-zinc-900 ring-1 ring-border"
      aria-label="Engine evaluation"
    >
      <div
        className="absolute inset-x-0 top-0 bg-zinc-100 transition-[height] duration-500 ease-out"
        style={{ height: `${whitePct}%` }}
      />
      <div className="pointer-events-none absolute inset-x-0 top-1/2 h-px -translate-y-px bg-white/10" />
      <span className="pointer-events-none absolute inset-x-0 top-1 text-center text-[8px] font-semibold uppercase tracking-wide text-zinc-400">
        W
      </span>
      <span className="pointer-events-none absolute inset-x-0 bottom-1 text-center text-[8px] font-semibold uppercase tracking-wide text-zinc-500">
        B
      </span>
      {loading && !analysis && (
        <div className="absolute inset-0 animate-pulse bg-accent/10" />
      )}
    </div>
  );
}

export function EngineReadout({
  analysis,
  loading,
}: {
  analysis: AnalysisResult | null;
  loading?: boolean;
}) {
  const cp = analysis?.evaluationCp ?? null;
  const mate = analysis?.evaluationMate ?? null;

  if (!analysis && !loading) return null;

  const evalText = analysis ? formatEval(cp, mate) : "…";
  const depthText = analysis ? `depth ${analysis.depth}` : "depth —";
  const bestText = analysis?.bestMoveSan ?? analysis?.bestMove ?? "—";
  const nodesText = analysis?.nodesSearched
    ? `${formatNodes(analysis.nodesSearched)} nodes`
    : null;

  return (
    <div
      className={cn(
        "flex flex-wrap items-center justify-center gap-x-2 gap-y-1 rounded-xl border border-border bg-card/60 px-3 py-2",
        "font-mono text-xs tabular-nums text-muted",
        loading && !analysis && "animate-pulse",
      )}
    >
      <span className="text-sm font-semibold text-foreground">{evalText}</span>
      <span className="text-border">|</span>
      <span>{depthText}</span>
      <span className="text-border">|</span>
      <span>
        best: <span className="text-foreground">{bestText}</span>
      </span>
      {nodesText && (
        <>
          <span className="text-border">|</span>
          <span>{nodesText}</span>
        </>
      )}
    </div>
  );
}

/** @deprecated horizontal eval — use EvalBarVertical */
export function EvalBar({ analysis }: { analysis: AnalysisResult | null }) {
  const cp = analysis?.evaluationCp ?? null;
  const mate = analysis?.evaluationMate ?? null;
  const whitePct = evalToPercent(cp, mate);

  return (
    <div className="flex items-stretch gap-3">
      <div className="relative h-36 w-4 shrink-0 overflow-hidden rounded-full bg-zinc-800 ring-1 ring-border">
        <div
          className="absolute bottom-0 w-full bg-zinc-100 transition-all duration-700 ease-out"
          style={{ height: `${whitePct}%` }}
        />
      </div>
      <div className="flex flex-col justify-center">
        <p className="text-[10px] font-medium uppercase tracking-widest text-muted">Eval</p>
        <p className="text-3xl font-semibold tabular-nums tracking-tight">
          {analysis ? formatEval(cp, mate) : "—"}
        </p>
        {analysis && (
          <p className="mt-1 text-xs text-muted">
            Best {analysis.bestMoveSan ?? analysis.bestMove} · d{analysis.depth}
          </p>
        )}
      </div>
    </div>
  );
}

export function EngineLines({ analysis }: { analysis: AnalysisResult | null }) {
  if (!analysis) {
    return <p className="py-6 text-center text-xs text-muted">Analysis will appear here</p>;
  }

  return (
    <div className="space-y-2">
      {analysis.lines.map((line, i) => (
        <div
          key={line.multipv}
          className={cn(
            "rounded-xl border border-border px-3 py-2.5",
            i === 0 && "border-accent/25 bg-accent/5",
          )}
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-muted">
              Line {line.multipv}
            </span>
            <span className="font-mono text-sm font-semibold tabular-nums">
              {formatEval(line.evalCp, line.evalMate)}
            </span>
          </div>
          <p className="mt-1 font-mono text-xs text-foreground">
            {line.san ?? line.move}
          </p>
          <p className="mt-0.5 font-mono text-xs text-muted">{line.pv.slice(0, 6).join(" ")}</p>
        </div>
      ))}
    </div>
  );
}

export function AnalysisPanel() {
  const analysis = useAppStore((s) => s.analysis);
  const analysisLoading = useAppStore((s) => s.analysisLoading);
  const boardReady = useAppStore((s) => s.boardReady);

  return (
    <div className="glass animate-fade-up space-y-4 rounded-2xl p-4">
      <div>
        <h3 className="text-sm font-semibold tracking-tight">Stockfish</h3>
        <p className="text-xs text-muted">
          {boardReady ? "Live engine analysis" : "Requires validated FEN"}
        </p>
      </div>
      <EvalBarVertical analysis={analysis} loading={analysisLoading} />
      <EngineReadout analysis={analysis} loading={analysisLoading} />
      <EngineLines analysis={analysis} />
    </div>
  );
}
