import type { SavedBoardSnapshot } from "@/types/boardSnapshot";

const STORAGE_KEY = "visionchess:board-snapshots";
const MAX_SNAPSHOTS = 50;

function readAll(): SavedBoardSnapshot[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as SavedBoardSnapshot[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeAll(snapshots: SavedBoardSnapshot[]): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshots.slice(0, MAX_SNAPSHOTS)));
}

export function listBoardSnapshots(): SavedBoardSnapshot[] {
  return readAll().sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
  );
}

export function upsertBoardSnapshot(snapshot: SavedBoardSnapshot): SavedBoardSnapshot[] {
  const existing = readAll();
  const index = existing.findIndex((item) => item.id === snapshot.id);
  const next =
    index >= 0
      ? existing.map((item, i) => (i === index ? snapshot : item))
      : [snapshot, ...existing];

  const trimmed = next
    .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
    .slice(0, MAX_SNAPSHOTS);

  writeAll(trimmed);
  return trimmed;
}

export function deleteBoardSnapshot(id: string): SavedBoardSnapshot[] {
  const next = readAll().filter((item) => item.id !== id);
  writeAll(next);
  return next;
}

export function getBoardSnapshot(id: string): SavedBoardSnapshot | undefined {
  return readAll().find((item) => item.id === id);
}

export function fenPlacement(fen: string): string {
  return fen.trim().split(/\s+/)[0] ?? fen;
}

export function shortFenLabel(fen: string, max = 28): string {
  const placement = fenPlacement(fen);
  if (placement.length <= max) return placement;
  return `${placement.slice(0, max - 1)}…`;
}
