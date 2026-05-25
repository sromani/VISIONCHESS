"use client";

import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";

interface UploadHeroProps {
  onSelect: () => void;
  onDrop: (file: File) => void;
  dragging: boolean;
  onDragChange: (dragging: boolean) => void;
  compact?: boolean;
}

export function UploadHero({
  onSelect,
  onDrop,
  dragging,
  onDragChange,
  compact = false,
}: UploadHeroProps) {
  if (compact) {
    return (
      <div
        className={cn(
          "flex w-full flex-col items-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-all",
          dragging
            ? "border-accent bg-accent-soft/20"
            : "border-border bg-card/40 hover:border-accent/40",
        )}
        onDragOver={(e) => {
          e.preventDefault();
          onDragChange(true);
        }}
        onDragLeave={() => onDragChange(false)}
        onDrop={(e) => {
          e.preventDefault();
          onDragChange(false);
          const file = e.dataTransfer.files[0];
          if (file?.type.startsWith("image/")) onDrop(file);
        }}
      >
        <Button size="lg" className="min-w-[200px]" onClick={onSelect}>
          Upload photo
        </Button>
        <p className="mt-3 text-xs text-muted">or drop an image here</p>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "animate-fade-up mx-auto flex w-full max-w-2xl flex-col items-center",
        "rounded-2xl border-2 border-dashed px-8 py-16 text-center transition-all duration-300",
        dragging
          ? "scale-[1.01] border-accent bg-accent-soft/20"
          : "border-border bg-card/40 hover:border-accent/40 hover:bg-card/60",
      )}
      onDragOver={(e) => {
        e.preventDefault();
        onDragChange(true);
      }}
      onDragLeave={() => onDragChange(false)}
      onDrop={(e) => {
        e.preventDefault();
        onDragChange(false);
        const file = e.dataTransfer.files[0];
        if (file?.type.startsWith("image/")) onDrop(file);
      }}
    >
      <div className="animate-pulse-ring mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-accent/10 text-3xl">
        ♟
      </div>

      <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
        Upload Chess Position
      </h1>
      <p className="mt-3 max-w-md text-sm leading-relaxed text-muted md:text-base">
        Drop a photo of any chess board. We&apos;ll detect the position, build an
        interactive board, and analyze it with Stockfish.
      </p>

      <Button size="lg" className="mt-8 min-w-[220px]" onClick={onSelect}>
        Choose image
      </Button>

      <p className="mt-4 text-xs text-muted">PNG, JPG, WebP · max 10 MB</p>
    </div>
  );
}
