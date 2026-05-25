"use client";

import { ThemeToggle } from "@/components/layout/ThemeToggle";

export function ProductHeader() {
  return (
    <header className="flex items-center justify-between px-4 py-4 md:px-8">
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-sm text-accent">
          ♞
        </div>
        <span className="text-sm font-semibold tracking-tight">VisionChess</span>
      </div>
      <ThemeToggle />
    </header>
  );
}
