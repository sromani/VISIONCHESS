"use client";

import { Button } from "@/components/ui/Button";
import { detectionsFromSquares, tryBuildFen } from "@/lib/chess/detections";
import { fenPlacement } from "@/lib/utils";
import { START_FEN } from "@/lib/pipeline";
import { useAppStore } from "@/store/appStore";
import type { DetectionResult } from "@/types";

export function PositionSetupPanel({
  detection,
  onClose,
}: {
  detection: DetectionResult | null;
  onClose?: () => void;
}) {
  const boardReady = useAppStore((s) => s.boardReady);
  const error = useAppStore((s) => s.error);
  const confirmDetectedPosition = useAppStore((s) => s.confirmDetectedPosition);
  const startAnalysisBoard = useAppStore((s) => s.startAnalysisBoard);
  const reset = useAppStore((s) => s.reset);
  const setFen = useAppStore((s) => s.setFen);
  const fen = useAppStore((s) => s.fen);

  if (boardReady) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-muted">
          Position is ready for interactive play. Use turn and castling controls below when
          available from the main screen.
        </p>
        {fen && (
          <code className="block break-all rounded-xl border border-border bg-card/50 px-3 py-2 font-mono text-[11px] text-foreground">
            {fen}
          </code>
        )}
        <Button variant="secondary" className="w-full touch-target" onClick={onClose}>
          Done
        </Button>
      </div>
    );
  }

  if (!detection) {
    return (
      <p className="text-sm text-muted">No detection data. Scan a board or start from the initial position.</p>
    );
  }

  const placement = fenPlacement(detection.fen || fen || "");
  const pieceCount = [...placement].filter((ch) => /[a-zA-Z]/.test(ch)).length;
  const detections = detectionsFromSquares(detection.squares ?? []);
  const previewFen = tryBuildFen(detections) ?? "";

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">
        Vision found pieces but the position is not valid for play (often missing kings or too few
        pieces). You can try to confirm anyway, use the starting position, or scan again.
      </p>

      <div className="rounded-xl border border-border bg-card/40 p-3">
        <p className="text-xs font-medium text-muted">Detected placement</p>
        <code className="mt-2 block break-all font-mono text-[11px] leading-relaxed text-foreground">
          {placement || "—"}
        </code>
        <p className="mt-2 text-xs text-muted">{pieceCount} pieces detected</p>
      </div>

      {previewFen && (
        <div className="rounded-xl border border-border bg-card/40 p-3">
          <p className="text-xs font-medium text-muted">Preview FEN (with defaults)</p>
          <code className="mt-2 block break-all font-mono text-[10px] leading-relaxed text-foreground">
            {previewFen}
          </code>
        </div>
      )}

      {error && (
        <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
          {error}
        </p>
      )}

      <div className="flex flex-col gap-2">
        <Button
          className="touch-target w-full"
          onClick={() => {
            if (confirmDetectedPosition()) onClose?.();
          }}
        >
          Confirm &amp; analyze
        </Button>
        <Button
          variant="secondary"
          className="touch-target w-full"
          onClick={() => {
            startAnalysisBoard(START_FEN, "Starting position");
            onClose?.();
          }}
        >
          Use starting position
        </Button>
        <Button
          variant="secondary"
          className="touch-target w-full"
          onClick={() => {
            reset();
            onClose?.();
          }}
        >
          Scan another photo
        </Button>
        {previewFen && (
          <Button
            variant="ghost"
            className="touch-target w-full"
            onClick={() => {
              setFen(previewFen);
              onClose?.();
            }}
          >
            Force preview FEN
          </Button>
        )}
      </div>
    </div>
  );
}
