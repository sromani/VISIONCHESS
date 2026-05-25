import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatEval(cp: number | null, mate: number | null): string {
  if (mate !== null) return `#${mate}`;
  if (cp === null) return "—";
  const pawns = cp / 100;
  return pawns > 0 ? `+${pawns.toFixed(1)}` : pawns.toFixed(1);
}

export function formatNodes(count: number | undefined): string {
  if (count === undefined) return "—";
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K`;
  return String(count);
}

export function evalToPercent(cp: number | null, mate: number | null): number {
  if (mate !== null) return mate > 0 ? 92 : 8;
  if (cp === null) return 50;
  const clamped = Math.max(-600, Math.min(600, cp));
  return 50 + (clamped / 600) * 42;
}

export function fenPlacement(fen: string): string {
  return fen.split(" ")[0] ?? fen;
}
