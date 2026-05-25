"use client";

import type { BootstrapProgress } from "@/lib/bootstrap/appBootstrap";
import { cn } from "@/lib/utils";

const SUBTITLE = "Offline Chess Vision";

export function AppLoadingScreen({
  steps,
  exiting = false,
  warning,
}: {
  steps: BootstrapProgress[];
  exiting?: boolean;
  warning?: string | null;
}) {
  const active = steps.find((s) => !s.done);
  const doneCount = steps.filter((s) => s.done).length;
  const progress = steps.length > 0 ? doneCount / steps.length : 0;

  return (
    <div
      className={cn(
        "app-loading-screen fixed inset-0 z-[100] flex flex-col items-center justify-center",
        exiting && "app-loading-screen--exit",
      )}
      role="status"
      aria-live="polite"
      aria-busy={!exiting}
    >
      <div className="app-loading-glow pointer-events-none absolute inset-0" aria-hidden />

      <div className="app-loading-content relative flex flex-col items-center px-8">
        <div className="app-loading-icon-wrap mb-8">
          <img
            src="/branding/app-icon-1024.png"
            alt=""
            width={88}
            height={88}
            className="app-loading-icon h-[88px] w-[88px] rounded-[22px] shadow-[0_0_48px_rgb(74_222_128/0.22)]"
            draggable={false}
          />
        </div>

        <h1 className="text-center text-[1.65rem] font-semibold tracking-tight text-foreground">
          VisionChess
        </h1>
        <p className="mt-2 text-center text-sm font-medium tracking-wide text-muted">
          {SUBTITLE}
        </p>

        <div className="mt-12 w-[min(100%,280px)]">
          <div className="relative h-[2px] overflow-hidden rounded-full bg-zinc-800/90">
            <div
              className="h-full rounded-full bg-gradient-to-r from-emerald-700/90 via-emerald-400 to-emerald-300/90 transition-[width] duration-500 ease-out"
              style={{ width: `${Math.max(10, progress * 100)}%` }}
            />
            <div
              className="app-loading-shimmer pointer-events-none absolute inset-y-0 left-0 w-1/3 rounded-full bg-white/20"
              aria-hidden
            />
          </div>
          <p className="mt-4 min-h-[1.25rem] text-center text-xs text-muted">
            {active?.label ?? (exiting ? "Ready" : "Starting…")}
            {!exiting && (
              <span className="mt-1 block text-[10px] text-zinc-500">
                YOLO (~100MB) loads on first scan, not at startup
              </span>
            )}
          </p>
          {warning && (
            <p className="mt-2 text-center text-[10px] text-amber-400/90">{warning}</p>
          )}
        </div>
      </div>
    </div>
  );
}
