"use client";

import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

const NAV = [
  { id: "upload", label: "Upload", icon: "↑" },
  { id: "board", label: "Board", icon: "♟", disabled: true },
  { id: "history", label: "History", icon: "◷", disabled: true },
];

export function Sidebar() {
  const phase = useAppStore((s) => s.phase);
  const reset = useAppStore((s) => s.reset);

  return (
    <aside className="flex h-full w-[220px] shrink-0 flex-col border-r border-border bg-sidebar px-4 py-5 lg:w-[240px]">
      <div className="mb-8 flex items-center gap-2.5 px-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-sm text-accent">
          ♞
        </div>
        <div>
          <p className="text-sm font-semibold tracking-tight">VisionChess</p>
          <p className="text-[10px] uppercase tracking-widest text-muted">Beta</p>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {NAV.map((item) => (
          <button
            key={item.id}
            type="button"
            disabled={item.disabled}
            onClick={item.id === "upload" ? reset : undefined}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition",
              item.id === "upload" && phase !== "idle"
                ? "bg-card text-foreground"
                : "text-muted hover:bg-card-hover hover:text-foreground",
              item.disabled && "cursor-not-allowed opacity-40",
            )}
          >
            <span className="text-base">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div className="mt-auto flex items-center justify-between px-2 pt-4">
        <p className="text-xs text-muted">API mode</p>
        <ThemeToggle />
      </div>
    </aside>
  );
}
