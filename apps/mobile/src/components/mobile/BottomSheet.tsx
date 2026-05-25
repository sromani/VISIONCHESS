import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

interface BottomSheetProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  className?: string;
}

export function BottomSheet({ open, onClose, title, children, className }: BottomSheetProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col justify-end">
      <button
        type="button"
        aria-label="Close panel"
        className="absolute inset-0 bg-black/50 backdrop-blur-[2px]"
        onClick={onClose}
      />
      <div
        className={cn(
          "animate-sheet-up relative max-h-[min(72dvh,560px)] rounded-t-2xl border border-border bg-card shadow-2xl",
          "safe-pb safe-px pt-4",
          className,
        )}
      >
        <div className="mx-auto mb-3 h-1 w-10 rounded-full bg-border" />
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            className="touch-target rounded-lg px-2 text-sm text-muted hover:text-foreground"
          >
            Done
          </button>
        </div>
        <div className="mobile-scroll max-h-[calc(min(72dvh,560px)-5rem)] pb-2">{children}</div>
      </div>
    </div>
  );
}
