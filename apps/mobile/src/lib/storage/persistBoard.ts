import {
  getActiveColor,
  parseCastlingRights,
} from "@/lib/chess/game";
import {
  listBoardSnapshots,
  upsertBoardSnapshot,
} from "@/lib/storage/boardSnapshots";
import type { SavedBoardSnapshot } from "@/types/boardSnapshot";

interface PersistBoardInput {
  id: string | null;
  fen: string;
  orientation: "white" | "black";
  title: string;
  imagePreview?: string;
  previous?: SavedBoardSnapshot | null;
}

export function buildBoardSnapshot(input: PersistBoardInput): SavedBoardSnapshot {
  const now = new Date().toISOString();
  const id = input.id ?? crypto.randomUUID();

  return {
    id,
    createdAt: input.previous?.createdAt ?? now,
    updatedAt: now,
    title: input.title,
    fen: input.fen,
    imagePreview: input.imagePreview ?? input.previous?.imagePreview,
    orientation: input.orientation,
    sideToMove: getActiveColor(input.fen),
    castlingRights: parseCastlingRights(input.fen),
  };
}

export function persistBoardSnapshot(input: PersistBoardInput): {
  snapshot: SavedBoardSnapshot;
  boards: SavedBoardSnapshot[];
} {
  const previous =
    input.previous ??
    (input.id ? listBoardSnapshots().find((board) => board.id === input.id) : undefined);

  const snapshot = buildBoardSnapshot({ ...input, previous });
  const boards = upsertBoardSnapshot(snapshot);
  return { snapshot, boards };
}
