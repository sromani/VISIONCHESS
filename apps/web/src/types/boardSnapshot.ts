import type { CastlingRights } from "@/lib/chess/game";

export interface SavedBoardSnapshot {
  id: string;
  createdAt: string;
  updatedAt: string;
  title: string;
  fen: string;
  imagePreview?: string;
  orientation: "white" | "black";
  sideToMove: "w" | "b";
  castlingRights: CastlingRights;
}
