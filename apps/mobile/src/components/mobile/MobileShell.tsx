import { useEffect } from "react";

import { RecentBoardsPanel } from "@/components/board/RecentBoardsPanel";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

export function MobileShell({ children }: { children: React.ReactNode }) {
  const hydrateSavedBoards = useAppStore((s) => s.hydrateSavedBoards);

  useEffect(() => {
    hydrateSavedBoards();
  }, [hydrateSavedBoards]);

  return (
    <div className="flex min-h-dvh flex-col bg-background">
      <header className="safe-pt safe-px sticky top-0 z-20 border-b border-border/80 bg-background/90 backdrop-blur-md">
        <div className="flex items-center justify-between py-3">
          <div>
            <p className="text-xs font-medium uppercase tracking-widest text-accent">VisionChess</p>
            <h1 className="text-lg font-semibold tracking-tight">Analysis Board</h1>
          </div>
          <span className="rounded-full border border-border px-2.5 py-1 text-[10px] font-medium uppercase tracking-wide text-muted">
            Offline
          </span>
        </div>
      </header>

      <main className={cn("mobile-scroll safe-px flex-1 pb-28 pt-4")}>{children}</main>

      <aside className="safe-pb safe-px fixed bottom-0 left-0 right-0 z-10 border-t border-border bg-card/95 backdrop-blur-md">
        <RecentBoardsPanel compact />
      </aside>
    </div>
  );
}
