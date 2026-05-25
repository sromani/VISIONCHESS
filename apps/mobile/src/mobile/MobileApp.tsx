import { useState } from "react";

import { AnalysisBoardSection } from "@/components/analysis/AnalysisBoardSection";
import { CastlingRightsPanel } from "@/components/board/CastlingRightsPanel";
import { MoveHistoryPanel } from "@/components/board/MoveHistoryPanel";
import { PositionSetupPanel } from "@/components/board/PositionSetupPanel";
import { TurnSelector } from "@/components/board/TurnSelector";
import { BottomSheet } from "@/components/mobile/BottomSheet";
import { CaptureActions } from "@/components/mobile/CaptureActions";
import { MobileShell } from "@/components/mobile/MobileShell";
import { EngineArrowsControls } from "@/components/analysis/EngineArrowsControls";
import { FenDisplay } from "@/components/fen/FenDisplay";
import { MinimalLoader } from "@/components/upload/MinimalLoader";
import { ScanStatusBanner } from "@/components/vision/ScanStatusBanner";
import { Button } from "@/components/ui/Button";
import { ErrorAlert } from "@/components/ui/ErrorAlert";
import { OFFLINE_MODE, VISION_LOCAL } from "@/lib/config";
import { captureBoardPhoto, pickBoardFromGallery } from "@/lib/native/camera";
import { START_FEN } from "@/lib/pipeline";
import { useAppStore } from "@/store/appStore";

/** Production Capacitor UI — no debug panels, no CV lab overlays. */
export default function MobileApp({ bootstrapWarning }: { bootstrapWarning?: string | null }) {
  const phase = useAppStore((s) => s.phase);
  const error = useAppStore((s) => s.error);
  const detection = useAppStore((s) => s.detection);
  const boardReady = useAppStore((s) => s.boardReady);
  const upload = useAppStore((s) => s.upload);
  const reset = useAppStore((s) => s.reset);
  const flipBoard = useAppStore((s) => s.flipBoard);
  const resetBoard = useAppStore((s) => s.resetBoard);
  const startAnalysisBoard = useAppStore((s) => s.startAnalysisBoard);

  const [sheet, setSheet] = useState<"moves" | "engine" | "setup" | null>(null);
  const [hoveredLine, setHoveredLine] = useState<import("@/types").EngineLine | null>(null);

  const isProcessing = ["uploading", "detecting", "classifying", "validating", "analyzing"].includes(
    phase,
  );

  async function handleFile(file: File | null) {
    if (!file) return;
    await upload(file);
  }

  return (
    <MobileShell>
      {bootstrapWarning && phase === "idle" && (
        <div className="mx-auto mb-4 w-full max-w-md rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200/90">
          {bootstrapWarning}
        </div>
      )}
      {phase === "idle" && !isProcessing && (
        <div className="mx-auto flex w-full max-w-md flex-col items-center gap-8 py-6">
          <div className="text-center">
            <h2 className="text-2xl font-semibold tracking-tight">Scan or analyze</h2>
            <p className="mt-2 text-sm text-muted">
              {VISION_LOCAL
                ? "Offline scan: local YOLO → FEN → Stockfish on device."
                : "Local Stockfish analysis on your device — no server required."}
            </p>
          </div>
          <CaptureActions
            offline={OFFLINE_MODE && !VISION_LOCAL}
            visionLocal={VISION_LOCAL}
            onCamera={() => {
              void captureBoardPhoto().then(handleFile);
            }}
            onGallery={() => {
              void pickBoardFromGallery().then(handleFile);
            }}
            onStartPosition={() => startAnalysisBoard(START_FEN, "Starting position")}
          />
        </div>
      )}

      {isProcessing && (
        <MinimalLoader
          label={
            VISION_LOCAL && phase === "detecting"
              ? "Loading YOLO + detecting pieces… first scan can take 1–3 min on phone"
              : "Reading position…"
          }
        />
      )}

      {phase === "error" && (
        <div className="mx-auto flex w-full max-w-md flex-col items-center gap-6 py-8">
          <ErrorAlert message={error ?? "Something went wrong"} />
          <div className="flex w-full flex-col gap-2">
            <Button onClick={reset}>Try again</Button>
            <Button variant="secondary" onClick={() => startAnalysisBoard(START_FEN)}>
              Start from initial position
            </Button>
          </div>
        </div>
      )}

      {phase === "ready" && (
        <div className="animate-fade-up mx-auto flex w-full max-w-lg flex-col items-center gap-4">
          {detection && <ScanStatusBanner detection={detection} boardReady={boardReady} />}

          <AnalysisBoardSection
            showInlineControls={false}
            engineHighlight={hoveredLine ? { from: hoveredLine.from, to: hoveredLine.to } : null}
          />

          <div className="grid w-full grid-cols-3 gap-2">
            <Button variant="secondary" size="sm" className="touch-target" onClick={() => setSheet("moves")}>
              Moves
            </Button>
            <Button variant="secondary" size="sm" className="touch-target" onClick={() => setSheet("engine")}>
              Engine
            </Button>
            <Button
              variant="secondary"
              size="sm"
              className="touch-target"
              onClick={() => {
                useAppStore.setState({ error: null });
                setSheet("setup");
              }}
            >
              Setup
            </Button>
          </div>

          <div className="flex w-full flex-wrap justify-center gap-2">
            {boardReady && (
              <>
                <Button variant="ghost" size="sm" className="touch-target" onClick={flipBoard}>
                  Flip
                </Button>
                <Button variant="ghost" size="sm" className="touch-target" onClick={resetBoard}>
                  Reset
                </Button>
              </>
            )}
            <Button variant="secondary" size="sm" className="touch-target" onClick={reset}>
              New board
            </Button>
          </div>
        </div>
      )}

      <BottomSheet open={sheet === "moves"} onClose={() => setSheet(null)} title="Move history">
        <MoveHistoryPanel />
      </BottomSheet>

      <BottomSheet open={sheet === "engine"} onClose={() => setSheet(null)} title="Engine">
        <EngineArrowsControls onHoverLine={setHoveredLine} />
        <FenDisplay />
      </BottomSheet>

      <BottomSheet open={sheet === "setup"} onClose={() => setSheet(null)} title="Position setup">
        <PositionSetupPanel detection={detection} onClose={() => setSheet(null)} />
        {boardReady && (
          <div className="mt-6 space-y-4 border-t border-border pt-4">
            <TurnSelector />
            <CastlingRightsPanel />
          </div>
        )}
      </BottomSheet>
    </MobileShell>
  );
}
