"use client";

import { useEffect } from "react";

import { analyzePosition, getStockfishEngine } from "@/lib/chess/stockfishEngine";
import {
  selectCurrentHistoryEntry,
  useAppStore,
} from "@/store/appStore";

const DEBOUNCE_MS = 250;

export function useStockfishAnalysis() {
  const boardReady = useAppStore((s) => s.boardReady);
  const fen = useAppStore((s) => selectCurrentHistoryEntry(s)?.fen ?? s.fen);
  const showEngineArrows = useAppStore((s) => s.showEngineArrows);
  const engineMultiPv = useAppStore((s) => s.engineMultiPv);
  const setAnalysis = useAppStore((s) => s.setAnalysis);
  const setAnalysisLoading = useAppStore((s) => s.setAnalysisLoading);

  const multiPv = showEngineArrows ? engineMultiPv : 1;

  useEffect(() => {
    if (!boardReady || !fen) {
      setAnalysis(null);
      setAnalysisLoading(false);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setAnalysisLoading(true);

      void analyzePosition(fen, {
        depth: 12,
        multiPv,
        signal: controller.signal,
        onUpdate: (analysis) => {
          if (!controller.signal.aborted) {
            setAnalysis(analysis);
          }
        },
      })
        .then((analysis) => {
          if (!controller.signal.aborted && analysis) {
            setAnalysis(analysis);
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) {
            setAnalysisLoading(false);
          }
        });
    }, DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
      getStockfishEngine().stop();
      setAnalysisLoading(false);
    };
  }, [fen, boardReady, multiPv, setAnalysis, setAnalysisLoading]);
}
