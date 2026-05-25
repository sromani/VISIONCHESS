"use client";

import {
  castlingRightsToFenField,
  inferCastlingRights,
  parseCastlingRights,
  type CastlingRights,
} from "@/lib/chess/game";
import { cn } from "@/lib/utils";
import {
  selectCanEditPosition,
  selectCurrentHistoryEntry,
  useAppStore,
} from "@/store/appStore";

const WHITE_OPTIONS: { key: keyof CastlingRights; label: string }[] = [
  { key: "whiteKingside", label: "O-O" },
  { key: "whiteQueenside", label: "O-O-O" },
];

const BLACK_OPTIONS: { key: keyof CastlingRights; label: string }[] = [
  { key: "blackKingside", label: "O-O" },
  { key: "blackQueenside", label: "O-O-O" },
];

export function CastlingRightsPanel() {
  const boardReady = useAppStore((s) => s.boardReady);
  const currentEntry = useAppStore(selectCurrentHistoryEntry);
  const canEdit = useAppStore(selectCanEditPosition);
  const setCastlingRights = useAppStore((s) => s.setCastlingRights);

  if (!boardReady || !currentEntry) return null;

  const current = parseCastlingRights(currentEntry.fen);
  const available = inferCastlingRights(currentEntry.fen);
  const fenField = castlingRightsToFenField(current);

  const toggle = (key: keyof CastlingRights) => {
    const next = { ...current, [key]: !current[key] };
    if (next[key] && !available[key]) return;
    setCastlingRights(next);
  };

  return (
    <div className="flex w-full max-w-[min(100%,560px)] flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-medium text-foreground">Castling rights</p>
        <code className="rounded-md bg-card/60 px-2 py-1 font-mono text-xs text-muted">
          {fenField}
        </code>
      </div>

      <div
        className={cn(
          "grid gap-3 rounded-xl border border-border bg-card/50 p-3",
          !canEdit && "opacity-60",
        )}
      >
        <CastlingRow
          title="White"
          options={WHITE_OPTIONS}
          current={current}
          available={available}
          disabled={!canEdit}
          onToggle={toggle}
        />
        <CastlingRow
          title="Black"
          options={BLACK_OPTIONS}
          current={current}
          available={available}
          disabled={!canEdit}
          onToggle={toggle}
        />
      </div>

      {!canEdit && (
        <p className="text-center text-xs text-muted">
          Reset the board to edit castling rights
        </p>
      )}
    </div>
  );
}

function CastlingRow({
  title,
  options,
  current,
  available,
  disabled,
  onToggle,
}: {
  title: string;
  options: { key: keyof CastlingRights; label: string }[];
  current: CastlingRights;
  available: CastlingRights;
  disabled: boolean;
  onToggle: (key: keyof CastlingRights) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="w-12 text-xs font-medium text-muted">{title}</span>
      {options.map(({ key, label }) => {
        const checked = current[key];
        const canEnable = available[key];
        const isDisabled = disabled || (!checked && !canEnable);

        return (
          <label
            key={key}
            className={cn(
              "inline-flex cursor-pointer items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-sm transition",
              checked && "border-accent/40 bg-accent/10",
              isDisabled && "cursor-not-allowed opacity-50",
            )}
          >
            <input
              type="checkbox"
              className="accent-accent"
              checked={checked}
              disabled={isDisabled}
              onChange={() => onToggle(key)}
            />
            <span>{label}</span>
          </label>
        );
      })}
    </div>
  );
}
