"use client";

import { useRef, useState } from "react";

import { AnalysisBoardSection } from "@/components/analysis/AnalysisBoardSection";
import { AnalysisPanel } from "@/components/analysis/AnalysisPanel";
import { BoardDebugPanel } from "@/components/board/BoardDebugPanel";
import { CastlingRightsPanel } from "@/components/board/CastlingRightsPanel";
import { MoveHistoryPanel } from "@/components/board/MoveHistoryPanel";
import { RecentBoardsPanel } from "@/components/board/RecentBoardsPanel";
import { TurnSelector } from "@/components/board/TurnSelector";
import { DetectionGallery } from "@/components/detection/DetectionGallery";
import { FenDisplay } from "@/components/fen/FenDisplay";
import { FenPanel } from "@/components/fen/FenPanel";
import { MinimalLoader } from "@/components/upload/MinimalLoader";
import { UploadHero } from "@/components/upload/UploadHero";
import { Button } from "@/components/ui/Button";
import { ErrorAlert } from "@/components/ui/ErrorAlert";
import { DEV_MODE } from "@/lib/config";
import { useAppStore } from "@/store/appStore";

export default function HomePage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const phase = useAppStore((s) => s.phase);
  const error = useAppStore((s) => s.error);
  const detection = useAppStore((s) => s.detection);
  const boardReady = useAppStore((s) => s.boardReady);
  const upload = useAppStore((s) => s.upload);
  const reset = useAppStore((s) => s.reset);
  const flipBoard = useAppStore((s) => s.flipBoard);
  const resetBoard = useAppStore((s) => s.resetBoard);
  const [dragging, setDragging] = useState(false);

  const isProcessing = ["uploading", "detecting", "classifying", "validating", "analyzing"].includes(
    phase,
  );

  const openFilePicker = () => inputRef.current?.click();

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6 md:py-10 lg:flex-row lg:items-start lg:gap-8">
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void upload(file);
          e.target.value = "";
        }}
      />

      <div className="flex min-w-0 flex-1 flex-col items-center">
        {phase === "idle" && (
          <div className="flex w-full max-w-md flex-col items-center gap-8 py-8 md:py-16">
            {!DEV_MODE && (
              <div className="text-center">
                <h1 className="text-2xl font-semibold tracking-tight">Scan a chess board</h1>
                <p className="mt-2 text-sm text-muted">
                  Upload a photo and play the detected position.
                </p>
              </div>
            )}
            <UploadHero
              dragging={dragging}
              onDragChange={setDragging}
              onSelect={openFilePicker}
              onDrop={(f) => void upload(f)}
              compact={!DEV_MODE}
            />
          </div>
        )}

        {isProcessing && <MinimalLoader />}

        {phase === "error" && (
          <div className="flex w-full max-w-md flex-col items-center gap-6 py-12">
            <ErrorAlert message={error ?? "Something went wrong"} />
            <div className="flex gap-3">
              <Button variant="secondary" onClick={reset}>
                Try again
              </Button>
              <Button onClick={openFilePicker}>Upload image</Button>
            </div>
          </div>
        )}

        {phase === "ready" && detection && (
          <div className="animate-fade-up flex w-full max-w-lg flex-col items-center gap-6">
            {!boardReady && !DEV_MODE && (
              <p className="text-center text-sm text-muted">
                Could not build a playable position. Try another photo.
              </p>
            )}

            <AnalysisBoardSection />

            <TurnSelector />

            <CastlingRightsPanel />

            <MoveHistoryPanel />

            <FenDisplay />

            <div className="flex flex-wrap items-center justify-center gap-2">
              {boardReady && (
                <>
                  <Button variant="ghost" size="sm" onClick={flipBoard}>
                    Flip
                  </Button>
                  <Button variant="ghost" size="sm" onClick={resetBoard}>
                    Reset
                  </Button>
                </>
              )}
              <Button variant="secondary" size="sm" onClick={openFilePicker}>
                New photo
              </Button>
            </div>

            {DEV_MODE && (
              <div className="mt-12 w-full max-w-7xl space-y-6 border-t border-border pt-12">
                <DetectionGallery detection={detection} />
                <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
                  <div />
                  <aside className="space-y-6">
                    <BoardDebugPanel />
                    <FenPanel />
                    <AnalysisPanel />
                  </aside>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <RecentBoardsPanel className="lg:sticky lg:top-6" />
    </div>
  );
}
