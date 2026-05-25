"use client";

import type { Color } from "chess.js";

import { getActiveColor } from "@/lib/chess/game";
import { cn } from "@/lib/utils";
import {
  selectCanChangeTurn,
  selectCurrentHistoryEntry,
  useAppStore,
} from "@/store/appStore";

const OPTIONS: { turn: Color; label: string }[] = [
  { turn: "w", label: "White" },
  { turn: "b", label: "Black" },
];

export function TurnSelector() {
  const boardReady = useAppStore((s) => s.boardReady);
  const currentEntry = useAppStore(selectCurrentHistoryEntry);
  const canChangeTurn = useAppStore(selectCanChangeTurn);
  const setActiveColor = useAppStore((s) => s.setActiveColor);

  if (!boardReady || !currentEntry) return null;

  const activeTurn = getActiveColor(currentEntry.fen);
  const turnLabel = activeTurn === "w" ? "White to move" : "Black to move";

  return (
    <div className="flex w-full max-w-[min(100%,560px)] flex-col items-center gap-3">
      <p className="text-sm font-medium text-foreground">{turnLabel}</p>

      <div
        className={cn(
          "inline-flex rounded-xl border border-border bg-card/50 p-1",
          !canChangeTurn && "opacity-60",
        )}
        role="group"
        aria-label="Side to move"
      >
        {OPTIONS.map(({ turn, label }) => {
          const selected = activeTurn === turn;
          return (
            <button
              key={turn}
              type="button"
              disabled={!canChangeTurn}
              onClick={() => setActiveColor(turn)}
              className={cn(
                "min-w-[5.5rem] rounded-lg px-4 py-2 text-sm font-medium transition",
                selected
                  ? "bg-accent text-zinc-950 shadow-sm dark:text-zinc-950"
                  : "text-muted hover:bg-card-hover hover:text-foreground",
                !canChangeTurn && "cursor-not-allowed",
              )}
            >
              {label}
            </button>
          );
        })}
      </div>

      {!canChangeTurn && (
        <p className="text-center text-xs text-muted">
          Reset the board to change side to move
        </p>
      )}
    </div>
  );
}
