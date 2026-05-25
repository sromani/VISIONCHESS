"use client";

import { useRef, useState } from "react";

import { AnalysisBoardSection } from "@/components/analysis/AnalysisBoardSection";
import { PendingPositionBoard } from "@/components/board/PendingPositionBoard";
import { BoardCropPreview } from "@/components/vision/BoardCropPreview";
import { ScanStatusBanner } from "@/components/vision/ScanStatusBanner";
import { Button } from "@/components/ui/Button";
import { ErrorAlert } from "@/components/ui/ErrorAlert";
import { MinimalLoader } from "@/components/upload/MinimalLoader";
import { UploadHero } from "@/components/upload/UploadHero";
import { START_FEN } from "@/lib/pipeline";
import { useAppStore } from "@/store/appStore";

export function WebScanWorkspace() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const phase = useAppStore((s) => s.phase);
  const error = useAppStore((s) => s.error);
  const detection = useAppStore((s) => s.detection);
  const boardReady = useAppStore((s) => s.boardReady);
  const upload = useAppStore((s) => s.upload);
  const reset = useAppStore((s) => s.reset);
  const startAnalysisBoard = useAppStore((s) => s.startAnalysisBoard);

  const isProcessing = ["uploading", "detecting", "classifying", "validating", "analyzing"].includes(
    phase,
  );

  function pickFile() {
    inputRef.current?.click();
  }

  function onFile(file: File | null) {
    if (!file?.type.startsWith("image/")) return;
    void upload(file);
  }

  return (
    <main className="flex min-h-0 flex-1 flex-col overflow-y-auto p-4 lg:p-6">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />

      {phase === "idle" && !isProcessing && (
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-6">
          <div>
            <h1 className="text-xl font-semibold">VisionChess CV Lab</h1>
            <p className="mt-1 text-sm text-muted">
              Web-only debug — upload a board photo. Inspect candidates in the left panel.
            </p>
          </div>
          <UploadHero
            onSelect={pickFile}
            onDrop={(f) => onFile(f)}
            dragging={dragging}
            onDragChange={setDragging}
          />
          <Button variant="secondary" onClick={() => startAnalysisBoard(START_FEN)}>
            Load starting position (smoke test)
          </Button>
        </div>
      )}

      {isProcessing && (
        <MinimalLoader label="Running vision pipeline… (first run loads ~100MB YOLO)" />
      )}

      {phase === "error" && (
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-4">
          <ErrorAlert message={error ?? "Detection failed"} />
          <Button onClick={reset}>Try another image</Button>
        </div>
      )}

      {phase === "ready" && detection && (
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
          <ScanStatusBanner detection={detection} boardReady={boardReady} />
          <BoardCropPreview detection={detection} />
          {boardReady ? (
            <AnalysisBoardSection showInlineControls />
          ) : (
            <PendingPositionBoard detection={detection} />
          )}
          <Button variant="secondary" onClick={reset}>
            New image
          </Button>
        </div>
      )}
    </main>
  );
}
