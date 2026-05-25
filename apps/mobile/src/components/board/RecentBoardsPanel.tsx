"use client";

import { Button } from "@/components/ui/Button";
import { shortFenLabel } from "@/lib/storage/boardSnapshots";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

export function RecentBoardsPanel({
  className,
  compact = false,
}: {
  className?: string;
  compact?: boolean;
}) {
  const savedBoards = useAppStore((s) => s.savedBoards);
  const currentSnapshotId = useAppStore((s) => s.currentSnapshotId);
  const loadSavedBoard = useAppStore((s) => s.loadSavedBoard);
  const deleteSavedBoard = useAppStore((s) => s.deleteSavedBoard);

  if (compact) {
    return (
      <div className={cn("py-2", className)}>
        <p className="mb-2 text-[10px] font-medium uppercase tracking-widest text-muted">
          Recent
        </p>
        <div className="mobile-scroll flex gap-2 overflow-x-auto pb-1">
          {savedBoards.length === 0 ? (
            <span className="text-xs text-muted">No saved boards yet</span>
          ) : (
            savedBoards.map((board) => {
              const active = board.id === currentSnapshotId;
              return (
                <button
                  key={board.id}
                  type="button"
                  onClick={() => loadSavedBoard(board.id)}
                  className={cn(
                    "flex shrink-0 items-center gap-2 rounded-xl border px-2 py-1.5 text-left transition",
                    active ? "border-accent/40 bg-accent/10" : "border-border bg-background/60",
                  )}
                >
                  <div className="h-9 w-9 overflow-hidden rounded-md bg-card">
                    {board.imagePreview ? (
                      <img src={board.imagePreview} alt="" className="h-full w-full object-cover" />
                    ) : (
                      <span className="flex h-full w-full items-center justify-center text-sm">♟</span>
                    )}
                  </div>
                  <span className="max-w-[88px] truncate text-xs font-medium">
                    {shortFenLabel(board.fen, 12)}
                  </span>
                </button>
              );
            })
          )}
        </div>
      </div>
    );
  }

  return (
    <aside
      className={cn(
        "flex w-full flex-col rounded-2xl border border-border bg-card/40 lg:w-72 lg:shrink-0",
        className,
      )}
    >
      <div className="border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold tracking-tight">Recent boards</h2>
        <p className="mt-0.5 text-xs text-muted">Last {savedBoards.length} saved positions</p>
      </div>

      <div className="max-h-[min(70vh,720px)] overflow-y-auto p-2">
        {savedBoards.length === 0 ? (
          <p className="px-2 py-6 text-center text-xs text-muted">
            Upload or edit a board to save it here automatically.
          </p>
        ) : (
          <ul className="space-y-2">
            {savedBoards.map((board) => {
              const active = board.id === currentSnapshotId;
              return (
                <li
                  key={board.id}
                  className={cn(
                    "rounded-xl border border-border bg-background/40 p-2 transition",
                    active && "border-accent/40 ring-1 ring-accent/20",
                  )}
                >
                  <div className="flex gap-2">
                    <div className="h-14 w-14 shrink-0 overflow-hidden rounded-lg bg-card">
                      {board.imagePreview ? (
                        <img src={board.imagePreview} alt="" className="h-full w-full object-cover" />
                      ) : (
                        <div className="flex h-full w-full items-center justify-center text-lg text-muted">
                          ♟
                        </div>
                      )}
                    </div>

                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium text-foreground">{board.title}</p>
                      <code className="mt-1 block truncate font-mono text-[10px] text-muted">
                        {shortFenLabel(board.fen, 24)}
                      </code>
                    </div>
                  </div>

                  <div className="mt-2 flex gap-1">
                    <Button
                      size="sm"
                      variant={active ? "primary" : "secondary"}
                      className="h-7 flex-1 px-2 text-xs"
                      onClick={() => loadSavedBoard(board.id)}
                    >
                      Open
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2 text-xs"
                      onClick={() => deleteSavedBoard(board.id)}
                    >
                      Delete
                    </Button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}
