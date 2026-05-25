"use client";

import { EvalBarVertical, EngineReadout } from "@/components/analysis/AnalysisPanel";
import { EngineArrowsControls } from "@/components/analysis/EngineArrowsControls";
import { InteractiveBoard } from "@/components/board/InteractiveBoard";
import { useStockfishAnalysis } from "@/hooks/useStockfishAnalysis";
import { selectIsAtLatestMove, useAppStore } from "@/store/appStore";
import type { EngineLine } from "@/types";

interface AnalysisBoardSectionProps {
  engineHighlight?: Pick<EngineLine, "from" | "to"> | null;
  onHoverLine?: (line: EngineLine | null) => void;
  showInlineControls?: boolean;
}

export function AnalysisBoardSection({
  engineHighlight = null,
  onHoverLine,
  showInlineControls = true,
}: AnalysisBoardSectionProps) {
  useStockfishAnalysis();

  const analysis = useAppStore((s) => s.analysis);
  const analysisLoading = useAppStore((s) => s.analysisLoading);
  const boardReady = useAppStore((s) => s.boardReady);
  const isAtLatest = useAppStore(selectIsAtLatestMove);

  if (!boardReady) {
    return <InteractiveBoard />;
  }

  return (
    <div className="flex w-full max-w-[min(100%,580px)] flex-col gap-3">
      {!isAtLatest && (
        <p className="text-center text-xs text-muted">Viewing earlier position — go to latest to play</p>
      )}
      <div className="flex items-stretch gap-2">
        <EvalBarVertical analysis={analysis} loading={analysisLoading} />
        <div className="min-w-0 flex-1">
          <InteractiveBoard embedded engineHighlight={engineHighlight} />
        </div>
      </div>
      <EngineReadout analysis={analysis} loading={analysisLoading} />
      {showInlineControls && <EngineArrowsControls onHoverLine={onHoverLine} />}
    </div>
  );
}
