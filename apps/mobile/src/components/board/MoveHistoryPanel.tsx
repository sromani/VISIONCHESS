"use client";

import { useMemo } from "react";

import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

export function MoveHistoryPanel() {
  const history = useAppStore((s) => s.history);
  const currentMoveIndex = useAppStore((s) => s.currentMoveIndex);
  const goToStart = useAppStore((s) => s.goToStart);
  const previousMove = useAppStore((s) => s.previousMove);
  const nextMove = useAppStore((s) => s.nextMove);
  const goToEnd = useAppStore((s) => s.goToEnd);
  const goToMove = useAppStore((s) => s.goToMove);

  const canGoBack = currentMoveIndex > 0;
  const canGoForward = currentMoveIndex < history.length - 1;

  const rows = useMemo(() => {
    const lines: { number?: number; moves: { san: string; index: number }[] }[] = [];
    let currentLine: { number?: number; moves: { san: string; index: number }[] } | null = null;

    for (let i = 1; i < history.length; i++) {
      const entry = history[i];
      if (!entry.san) continue;

      if (entry.move?.color === "w") {
        if (currentLine) lines.push(currentLine);
        currentLine = {
          number: lines.filter((line) => line.number !== undefined).length + 1,
          moves: [{ san: entry.san, index: i }],
        };
        continue;
      }

      if (currentLine) {
        currentLine.moves.push({ san: entry.san, index: i });
        lines.push(currentLine);
        currentLine = null;
      } else {
        lines.push({ moves: [{ san: entry.san, index: i }] });
      }
    }

    if (currentLine) lines.push(currentLine);
    return lines;
  }, [history]);

  if (history.length <= 1) {
    return (
      <div className="w-full max-w-[min(100%,560px)] space-y-3">
        <div className="flex items-center justify-center gap-1">
          <NavButton label="Start" disabled onClick={goToStart}>
            |◀
          </NavButton>
          <NavButton label="Previous" disabled onClick={previousMove}>
            ◀
          </NavButton>
          <NavButton label="Next" disabled onClick={nextMove}>
            ▶
          </NavButton>
          <NavButton label="End" disabled onClick={goToEnd}>
            ▶|
          </NavButton>
        </div>
        <p className="text-center text-xs text-muted">No moves yet</p>
      </div>
    );
  }

  return (
    <div className="w-full max-w-[min(100%,560px)] space-y-3">
      <div className="flex items-center justify-center gap-1">
        <NavButton label="Start" disabled={!canGoBack} onClick={goToStart}>
          |◀
        </NavButton>
        <NavButton label="Previous" disabled={!canGoBack} onClick={previousMove}>
          ◀
        </NavButton>
        <NavButton label="Next" disabled={!canGoForward} onClick={nextMove}>
          ▶
        </NavButton>
        <NavButton label="End" disabled={!canGoForward} onClick={goToEnd}>
          ▶|
        </NavButton>
      </div>

      <input
        type="range"
        min={0}
        max={history.length - 1}
        value={currentMoveIndex}
        onChange={(e) => goToMove(Number(e.target.value))}
        className="w-full accent-accent"
        aria-label="Move timeline"
      />

      <div className="rounded-xl border border-border bg-card/50 px-3 py-2 font-mono text-sm leading-relaxed">
        {rows.map((row, rowIndex) => (
          <span key={rowIndex} className="mr-3 inline-block whitespace-nowrap">
            {row.number !== undefined && <span className="text-muted">{row.number}.</span>}{" "}
            {row.moves.map((move, moveIndex) => (
              <span key={move.index}>
                {moveIndex > 0 ? " " : row.number === undefined ? "" : ""}
                <MoveToken
                  san={move.san}
                  active={currentMoveIndex === move.index}
                  onClick={() => goToMove(move.index)}
                />
              </span>
            ))}
          </span>
        ))}
      </div>
    </div>
  );
}

function NavButton({
  children,
  label,
  disabled,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <Button
      variant="ghost"
      size="sm"
      disabled={disabled}
      onClick={onClick}
      className="min-w-[2.5rem] px-2 font-mono text-base"
      aria-label={label}
    >
      {children}
    </Button>
  );
}

function MoveToken({
  san,
  active,
  onClick,
}: {
  san: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded px-1.5 py-0.5 transition",
        active
          ? "bg-accent/25 font-semibold text-accent"
          : "text-foreground hover:bg-card-hover",
      )}
    >
      {san}
    </button>
  );
}
