"use client";

import { cn } from "@/lib/utils";
import { isOccupiedDetection, PieceDetection } from "@/lib/chess/detections";

const LABEL_SHORT: Record<string, string> = {
  empty: "·",
  white_pawn: "P",
  white_knight: "N",
  white_bishop: "B",
  white_rook: "R",
  white_queen: "Q",
  white_king: "K",
  black_pawn: "p",
  black_knight: "n",
  black_bishop: "b",
  black_rook: "r",
  black_queen: "q",
  black_king: "k",
};

export function DetectionBoardGrid({
  detections,
  highlightSquares,
  className,
}: {
  detections: PieceDetection[];
  highlightSquares?: Set<string>;
  className?: string;
}) {
  const bySquare = new Map(detections.map((d) => [d.square, d]));

  return (
    <div className={cn("inline-grid grid-cols-8 gap-px rounded-lg bg-border/40 p-1", className)}>
      {Array.from({ length: 8 }, (_, row) =>
        Array.from({ length: 8 }, (_, col) => {
          const name = `${String.fromCharCode(97 + col)}${8 - row}`;
          const det = bySquare.get(name);
          const occupied = det ? isOccupiedDetection(det) : false;
          const label = occupied ? det!.piece : "empty";
          const sym = LABEL_SHORT[label] ?? "?";
          const conf = occupied ? det!.confidence : undefined;
          const highlighted = highlightSquares?.has(name);

          return (
            <div
              key={name}
              className={cn(
                "flex h-9 w-9 flex-col items-center justify-center rounded-sm text-[10px] sm:h-10 sm:w-10",
                (row + col) % 2 === 0 ? "bg-[#ebecd0] text-zinc-900" : "bg-[#779556] text-zinc-100",
                highlighted && "ring-2 ring-amber-400 ring-inset",
              )}
              title={`${name} ${label}${conf != null ? ` ${(conf * 100).toFixed(0)}%` : ""}`}
            >
              <span className="font-bold">{sym}</span>
              {conf != null && (
                <span className={cn("text-[7px]", (row + col) % 2 === 0 ? "text-zinc-600" : "text-zinc-200")}>
                  {(conf * 100).toFixed(0)}
                </span>
              )}
            </div>
          );
        }),
      )}
    </div>
  );
}
