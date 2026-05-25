"use client";

import { useMemo, useState } from "react";
import { Chessboard } from "react-chessboard";

import { Button } from "@/components/ui/Button";
import {
  BoardOrientation,
  buildFen,
  detectionsToBoardPosition,
  lowConfidenceSquares,
  PieceDetection,
} from "@/lib/chess/detections";
import {
  buildCustomPieces,
  LICHESS_BOARD,
  PIECE_SET_OPTIONS,
  PieceSetId,
} from "@/lib/chess/pieceSets";
import { fenPlacement } from "@/lib/utils";
import { cn } from "@/lib/utils";

export function SyntheticBoard({
  detections,
  fen: fenOverride,
  orientation: initialOrientation = "white",
  showControls = true,
  showFen = true,
  showHighlights = true,
  highlightThreshold = 0.5,
  draggable = false,
  className,
  boardId = "SyntheticBoard",
}: {
  detections?: PieceDetection[];
  fen?: string;
  orientation?: BoardOrientation;
  showControls?: boolean;
  showFen?: boolean;
  showHighlights?: boolean;
  highlightThreshold?: number;
  draggable?: boolean;
  className?: string;
  boardId?: string;
}) {
  const [orientation, setOrientation] = useState<BoardOrientation>(initialOrientation);
  const [pieceSet, setPieceSet] = useState<PieceSetId>("cburnett");
  const [highlightsOn, setHighlightsOn] = useState(showHighlights);

  const computedFen = useMemo(
    () => (detections?.length ? buildFen(detections, { orientation }) : undefined),
    [detections, orientation],
  );
  const fen = fenOverride ?? computedFen ?? "8/8/8/8/8/8/8/8 w - - 0 1";
  const placement = fenPlacement(fen);

  const position = useMemo(() => {
    if (detections?.length) return detectionsToBoardPosition(detections);
    return fen;
  }, [detections, fen]);

  const customPieces = useMemo(() => buildCustomPieces(pieceSet), [pieceSet]);

  const customSquareStyles = useMemo((): Record<string, Record<string, string>> | undefined => {
    if (!highlightsOn || !detections?.length) return undefined;
    const squares = lowConfidenceSquares(detections, highlightThreshold);
    if (!squares.length) return undefined;
    const styles: Record<string, Record<string, string>> = {};
    for (const sq of squares) {
      styles[sq] = {
        backgroundColor: "rgba(251, 191, 36, 0.45)",
      };
    }
    return styles;
  }, [detections, highlightThreshold, highlightsOn]);

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {showControls && (
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => setOrientation((o) => (o === "white" ? "black" : "white"))}>
            Flip
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setHighlightsOn((v) => !v)}>
            {highlightsOn ? "Hide highlights" : "Show highlights"}
          </Button>
          <div className="flex gap-1">
            {PIECE_SET_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setPieceSet(opt.id)}
                className={cn(
                  "rounded-md px-2 py-1 text-[10px] font-medium transition-colors",
                  pieceSet === opt.id
                    ? "bg-accent/15 text-accent"
                    : "text-muted hover:bg-card hover:text-foreground",
                )}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="overflow-hidden rounded-xl border border-border shadow-[0_8px_32px_rgb(0_0_0/0.18)]">
        <div className="mx-auto aspect-square w-full">
          <Chessboard
            id={boardId}
            position={position}
            boardOrientation={orientation}
            arePiecesDraggable={draggable}
            animationDuration={120}
            showBoardNotation
            customPieces={customPieces}
            customBoardStyle={{ borderRadius: 0 }}
            customLightSquareStyle={LICHESS_BOARD.light}
            customDarkSquareStyle={LICHESS_BOARD.dark}
            customSquareStyles={customSquareStyles}
          />
        </div>
      </div>

      {showFen && (
        <p className="break-all font-mono text-[10px] text-muted" title="FEN placement">
          {placement}
        </p>
      )}
    </div>
  );
}
