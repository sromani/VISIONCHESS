"use client";

import { useCallback, useMemo, useState } from "react";
import { Chessboard } from "react-chessboard";
import type { Square as BoardSquare } from "react-chessboard/dist/chessboard/types";
import type { Square as ChessSquare } from "chess.js";

import { Button } from "@/components/ui/Button";
import {
  buildSquareStyles,
  tryCreateGame,
  findLegalMove,
  getPromotionMoves,
  isSquareSelectable,
  promotionFromBoardPiece,
  type BoardPieceCode,
  type PromotionPiece,
} from "@/lib/chess/game";
import { buildCustomPieces, LICHESS_BOARD } from "@/lib/chess/pieceSets";
import { buildEngineArrows, type BoardArrow, ENGINE_SQUARE_BEST_GLOW, ENGINE_SQUARE_BEST_GLOW_SHADOW, ENGINE_SQUARE_HOVER_FROM, ENGINE_SQUARE_HOVER_TO } from "@/lib/chess/engineArrows";
import { cn } from "@/lib/utils";
import { selectIsAtLatestMove, useAppStore } from "@/store/appStore";

import type { EngineLine } from "@/types";

const PROMOTION_LABELS: Record<PromotionPiece, string> = {
  q: "Queen",
  r: "Rook",
  b: "Bishop",
  n: "Knight",
};

export function InteractiveBoard({
  embedded = false,
  engineHighlight = null,
}: {
  embedded?: boolean;
  engineHighlight?: Pick<EngineLine, "from" | "to"> | null;
}) {
  const boardReady = useAppStore((s) => s.boardReady);
  const orientation = useAppStore((s) => s.orientation);
  const history = useAppStore((s) => s.history);
  const currentMoveIndex = useAppStore((s) => s.currentMoveIndex);
  const makeMove = useAppStore((s) => s.makeMove);
  const analysis = useAppStore((s) => s.analysis);
  const showEngineArrows = useAppStore((s) => s.showEngineArrows);
  const engineMultiPv = useAppStore((s) => s.engineMultiPv);
  const isAtLatest = useAppStore(selectIsAtLatestMove);

  const currentEntry = useMemo(
    () => history[currentMoveIndex] ?? null,
    [history, currentMoveIndex],
  );
  const boardFen = currentEntry?.fen ?? null;
  const lastMove = currentEntry?.lastMove ?? null;

  const [selectedSquare, setSelectedSquare] = useState<string | null>(null);
  const [promotionSquares, setPromotionSquares] = useState<{
    from: string;
    to: string;
  } | null>(null);

  const customPieces = useMemo(() => buildCustomPieces("cburnett"), []);

  const game = useMemo(() => (boardFen ? tryCreateGame(boardFen) : null), [boardFen]);

  const promotionOptions = useMemo(() => {
    if (!game || !promotionSquares || !isAtLatest) return [] as PromotionPiece[];
    return getPromotionMoves(
      game,
      promotionSquares.from as ChessSquare,
      promotionSquares.to as ChessSquare,
    )
      .map((move) => move.promotion)
      .filter((piece): piece is PromotionPiece => Boolean(piece));
  }, [game, promotionSquares, isAtLatest]);

  const customArrows = useMemo((): BoardArrow[] => {
    if (!showEngineArrows || !analysis?.lines.length) return [];
    return buildEngineArrows(analysis.lines, engineMultiPv);
  }, [showEngineArrows, analysis, engineMultiPv]);

  const customSquareStyles = useMemo(() => {
    if (!game) return {};
    const styles = buildSquareStyles(game, isAtLatest ? selectedSquare : null, lastMove);

    const bestLine = showEngineArrows ? analysis?.lines[0] : null;
    if (bestLine && !engineHighlight) {
      const from = bestLine.from as BoardSquare;
      const to = bestLine.to as BoardSquare;
      styles[from] = {
        ...styles[from],
        backgroundColor: ENGINE_SQUARE_BEST_GLOW,
        boxShadow: ENGINE_SQUARE_BEST_GLOW_SHADOW,
      };
      styles[to] = {
        ...styles[to],
        backgroundColor: ENGINE_SQUARE_BEST_GLOW,
      };
    }

    if (engineHighlight?.from) {
      const from = engineHighlight.from as BoardSquare;
      styles[from] = {
        ...styles[from],
        backgroundColor: ENGINE_SQUARE_HOVER_FROM,
      };
    }
    if (engineHighlight?.to) {
      const to = engineHighlight.to as BoardSquare;
      styles[to] = {
        ...styles[to],
        backgroundColor: ENGINE_SQUARE_HOVER_TO,
      };
    }
    return styles;
  }, [game, selectedSquare, lastMove, isAtLatest, engineHighlight, showEngineArrows, analysis]);

  const clearSelection = useCallback(() => {
    setSelectedSquare(null);
    setPromotionSquares(null);
  }, []);

  const attemptMove = useCallback(
    (from: string, to: string, promotion?: PromotionPiece) => {
      if (!isAtLatest) return false;
      const ok = makeMove(from, to, promotion);
      if (ok) clearSelection();
      return ok;
    },
    [makeMove, clearSelection, isAtLatest],
  );

  const onDrop = useCallback(
    (source: string, target: string) => attemptMove(source, target),
    [attemptMove],
  );

  const onPromotionPieceSelect = useCallback(
    (
      piece?: BoardPieceCode,
      promoteFromSquare?: string,
      promoteToSquare?: string,
    ) => {
      if (!piece || !promoteFromSquare || !promoteToSquare) return false;
      return attemptMove(
        promoteFromSquare,
        promoteToSquare,
        promotionFromBoardPiece(piece),
      );
    },
    [attemptMove],
  );

  const onSquareClick = useCallback(
    (square: string) => {
      if (!game || !boardFen || !isAtLatest) return;

      if (!selectedSquare) {
        if (isSquareSelectable(game, square as ChessSquare)) {
          setSelectedSquare(square);
        }
        return;
      }

      if (selectedSquare === square) {
        clearSelection();
        return;
      }

      const legal = findLegalMove(game, selectedSquare as ChessSquare, square as ChessSquare);
      if (!legal) {
        if (isSquareSelectable(game, square as ChessSquare)) {
          setSelectedSquare(square);
        } else {
          clearSelection();
        }
        return;
      }

      if (legal.isPromotion()) {
        setPromotionSquares({ from: selectedSquare, to: square });
        return;
      }

      attemptMove(selectedSquare, square);
    },
    [game, boardFen, selectedSquare, attemptMove, clearSelection, isAtLatest],
  );

  const isDraggablePiece = useCallback(
    ({ sourceSquare }: { piece: BoardPieceCode; sourceSquare: string }) => {
      if (!game || !isAtLatest) return false;
      return isSquareSelectable(game, sourceSquare as ChessSquare);
    },
    [game, isAtLatest],
  );

  if (!boardReady || !boardFen || !game || !currentEntry) {
    return null;
  }

  const boardShell = (
    <>
      {!embedded && !isAtLatest && (
        <p className="text-center text-xs text-muted">Viewing earlier position — go to latest to play</p>
      )}

      <div
        className={cn(
          "overflow-hidden rounded-2xl border border-border shadow-[0_8px_40px_rgb(0_0_0/0.12)]",
          "dark:shadow-[0_8px_40px_rgb(0_0_0/0.45)]",
        )}
      >
        <div className="aspect-square w-full">
          <Chessboard
            id="VisionBoard"
            position={boardFen}
            onPieceDrop={onDrop}
            onSquareClick={onSquareClick}
            onPromotionPieceSelect={onPromotionPieceSelect}
            isDraggablePiece={isDraggablePiece}
            boardOrientation={orientation}
            animationDuration={180}
            arePiecesDraggable={isAtLatest}
            areArrowsAllowed={false}
            customArrows={customArrows}
            showBoardNotation
            customPieces={customPieces}
            customSquareStyles={customSquareStyles}
            customBoardStyle={{ borderRadius: 0 }}
            customDarkSquareStyle={LICHESS_BOARD.dark}
            customLightSquareStyle={LICHESS_BOARD.light}
          />
        </div>
      </div>

      {isAtLatest && promotionSquares && promotionOptions.length > 0 && (
        <div className="rounded-xl border border-border bg-card/80 p-3 text-center">
          <p className="text-xs font-medium text-foreground">Promote to</p>
          <div className="mt-2 flex flex-wrap justify-center gap-2">
            {promotionOptions.map((piece) => (
              <Button
                key={piece}
                size="sm"
                variant="secondary"
                onClick={() =>
                  attemptMove(promotionSquares.from, promotionSquares.to, piece)
                }
              >
                {PROMOTION_LABELS[piece]}
              </Button>
            ))}
            <Button size="sm" variant="ghost" onClick={clearSelection}>
              Cancel
            </Button>
          </div>
        </div>
      )}
    </>
  );

  if (embedded) {
    return boardShell;
  }

  return (
    <div className="flex w-full max-w-[min(100%,560px)] flex-col gap-3">
      {boardShell}
    </div>
  );
}
