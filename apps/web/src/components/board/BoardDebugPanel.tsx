"use client";

import { useMemo } from "react";

import { getGameStatus } from "@/lib/chess/game";
import { cn } from "@/lib/utils";
import { selectCurrentHistoryEntry, useAppStore } from "@/store/appStore";

function StatusBadge({
  active,
  label,
  tone = "neutral",
}: {
  active: boolean;
  label: string;
  tone?: "neutral" | "warn" | "danger" | "ok";
}) {
  if (!active) return null;

  return (
    <span
      className={cn(
        "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
        tone === "danger" && "bg-red-500/15 text-red-400",
        tone === "warn" && "bg-amber-500/15 text-amber-400",
        tone === "ok" && "bg-emerald-500/15 text-emerald-400",
        tone === "neutral" && "bg-muted/20 text-muted",
      )}
    >
      {label}
    </span>
  );
}

export function BoardDebugPanel() {
  const boardReady = useAppStore((s) => s.boardReady);
  const currentMoveIndex = useAppStore((s) => s.currentMoveIndex);
  const history = useAppStore((s) => s.history);
  const currentEntry = useAppStore(selectCurrentHistoryEntry);

  const status = useMemo(
    () => (currentEntry ? getGameStatus(currentEntry.fen) : null),
    [currentEntry],
  );

  if (!boardReady || !currentEntry || !status) {
    return null;
  }

  const playedMoves = history.slice(1).map((entry) => entry.san).filter(Boolean);

  return (
    <div className="glass animate-fade-up rounded-2xl p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold tracking-tight">Board debug</h3>
        <StatusBadge active={status.inCheck} label="Check" tone="warn" />
        <StatusBadge active={status.isCheckmate} label="Checkmate" tone="danger" />
        <StatusBadge active={status.isStalemate} label="Stalemate" tone="neutral" />
        <StatusBadge active={status.isDraw && !status.isStalemate} label="Draw" tone="neutral" />
        <StatusBadge
          active={!status.isGameOver}
          label={`${status.turnLabel} to move`}
          tone="ok"
        />
      </div>

      <dl className="mt-4 space-y-3 text-xs">
        <div>
          <dt className="font-medium text-muted">Current FEN</dt>
          <dd className="mt-1 break-all rounded-lg bg-background/60 p-2 font-mono leading-relaxed">
            {currentEntry.fen}
          </dd>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <dt className="font-medium text-muted">Move index</dt>
            <dd className="mt-1 font-mono">
              {currentMoveIndex} / {history.length - 1}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-muted">Moves played</dt>
            <dd className="mt-1 font-mono">{playedMoves.length}</dd>
          </div>
        </div>

        {currentEntry.lastMove && (
          <div>
            <dt className="font-medium text-muted">Last move</dt>
            <dd className="mt-1 font-mono">
              {currentEntry.lastMove.from} → {currentEntry.lastMove.to}
              {currentEntry.san ? ` (${currentEntry.san})` : ""}
            </dd>
          </div>
        )}

        <div>
          <dt className="font-medium text-muted">Move list</dt>
          <dd className="mt-1 min-h-[1.5rem] rounded-lg bg-background/60 p-2 font-mono leading-relaxed">
            {playedMoves.length > 0 ? playedMoves.join(" ") : "—"}
          </dd>
        </div>

        <div>
          <dt className="font-medium text-muted">PGN</dt>
          <dd className="mt-1 whitespace-pre-wrap break-words rounded-lg bg-background/60 p-2 font-mono leading-relaxed">
            {currentEntry.pgn || "—"}
          </dd>
        </div>
      </dl>
    </div>
  );
}
