"use client";

import type { ReactElement } from "react";

import type { BoardPiece } from "@/lib/chess/detections";

type CustomPieceFn = (args: {
  isDragging: boolean;
  squareWidth: number;
  square?: string;
}) => ReactElement;

type CustomPieces = Partial<Record<BoardPiece, CustomPieceFn>>;

export type PieceSetId = "cburnett" | "merida" | "alpha";

export const PIECE_SET_OPTIONS: { id: PieceSetId; label: string }[] = [
  { id: "cburnett", label: "Cburnett" },
  { id: "merida", label: "Merida" },
  { id: "alpha", label: "Alpha" },
];

const ALL_PIECES: BoardPiece[] = [
  "wP",
  "wN",
  "wB",
  "wR",
  "wQ",
  "wK",
  "bP",
  "bN",
  "bB",
  "bR",
  "bQ",
  "bK",
];

function lichessPieceUrl(set: PieceSetId, piece: BoardPiece): string {
  return `https://lichess1.org/assets/piece/${set}/${piece}.svg`;
}

function renderPiece(set: PieceSetId, piece: BoardPiece): CustomPieceFn {
  return ({ squareWidth, isDragging }) => (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={lichessPieceUrl(set, piece)}
      width={squareWidth}
      height={squareWidth}
      alt=""
      draggable={false}
      style={{
        display: "block",
        opacity: isDragging ? 0.85 : 1,
        filter: isDragging ? "drop-shadow(0 6px 10px rgba(0,0,0,0.35))" : undefined,
      }}
    />
  );
}

export function buildCustomPieces(set: PieceSetId): CustomPieces {
  const pieces: CustomPieces = {};
  for (const piece of ALL_PIECES) {
    pieces[piece] = renderPiece(set, piece);
  }
  return pieces;
}

/** Lichess / chess.com style board colors */
export const LICHESS_BOARD = {
  light: { backgroundColor: "#ebecd0" },
  dark: { backgroundColor: "#779556" },
} as const;
