"use client";

import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { Button } from "@/components/ui/Button";
import { useAppStore } from "@/store/appStore";

export function MobileHeader() {
  const phase = useAppStore((s) => s.phase);
  const reset = useAppStore((s) => s.reset);

  return (
    <header className="flex items-center justify-between border-b border-border px-4 py-3 md:hidden">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-sm text-accent">
          ♞
        </div>
        <span className="text-sm font-semibold tracking-tight">VisionChess</span>
      </div>
      <div className="flex items-center gap-2">
        {phase !== "idle" && (
          <Button variant="ghost" size="sm" onClick={reset}>
            New
          </Button>
        )}
        <ThemeToggle />
      </div>
    </header>
  );
}
