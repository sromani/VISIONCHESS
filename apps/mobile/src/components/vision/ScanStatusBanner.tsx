"use client";

import type { DetectionResult } from "@/types";

export function ScanStatusBanner({
  detection,
  boardReady,
}: {
  detection: DetectionResult;
  boardReady: boolean;
}) {
  const status = detection.metadata?.localization_status as string | undefined;
  const boardFound = detection.metadata?.board_found === true;
  const custom = detection.metadata?.user_message as string | undefined;

  if (custom) {
    const isNotFound = status === "not_found" || !boardFound;
    return (
      <div
        className={
          isNotFound
            ? "w-full rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-100"
            : "w-full rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100"
        }
      >
        {custom}
      </div>
    );
  }

  if (!boardReady && boardFound) {
    return (
      <div className="w-full rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
        Tablero detectado pero la posición no es válida para jugar. Abrí{" "}
        <strong className="font-medium">Setup</strong> para confirmar o usá la posición inicial.
      </div>
    );
  }

  if (!boardReady && !boardFound) {
    return (
      <div className="w-full rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-100">
        No se encontró tablero en la foto.
      </div>
    );
  }

  return null;
}
