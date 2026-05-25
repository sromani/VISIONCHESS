import { useState } from "react";

import { AnalysisBoardSection } from "@/components/analysis/AnalysisBoardSection";
import { CastlingRightsPanel } from "@/components/board/CastlingRightsPanel";
import { MoveHistoryPanel } from "@/components/board/MoveHistoryPanel";
import { TurnSelector } from "@/components/board/TurnSelector";
import { BottomSheet } from "@/components/mobile/BottomSheet";
import { CaptureActions } from "@/components/mobile/CaptureActions";
import { MobileShell } from "@/components/mobile/MobileShell";
import { EngineArrowsControls } from "@/components/analysis/EngineArrowsControls";
import { FenDisplay } from "@/components/fen/FenDisplay";
import { MinimalLoader } from "@/components/upload/MinimalLoader";
import { Button } from "@/components/ui/Button";
import { ErrorAlert } from "@/components/ui/ErrorAlert";
import { OFFLINE_MODE } from "@/lib/config";
import { captureBoardPhoto, pickBoardFromGallery } from "@/lib/native/camera";
import { START_FEN } from "@/lib/pipeline";
import { useAppStore } from "@/store/appStore";

export default function App() {
  const phase = useAppStore((s) => s.phase);
  const error = useAppStore((s) => s.error);
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
    if (file) await upload(file);
  }

  return (
    <MobileShell>
      {phase === "idle" && (
        <div className="mx-auto flex w-full max-w-md flex-col items-center gap-8 py-6">
          <div className="text-center">
            <h2 className="text-2xl font-semibold tracking-tight">Scan or analyze</h2>
            <p className="mt-2 text-sm text-muted">
              Local Stockfish analysis on your device — no server required.
            </p>
          </div>
          <CaptureActions
            offline={OFFLINE_MODE}
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

      {isProcessing && <MinimalLoader />}

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
            <Button variant="secondary" size="sm" className="touch-target" onClick={() => setSheet("setup")}>
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
        <div className="space-y-4">
          <TurnSelector />
          <CastlingRightsPanel />
        </div>
      </BottomSheet>
    </MobileShell>
  );
}
