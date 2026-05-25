import { create } from "zustand";

import { getDetectionMode, type DetectionMode } from "@/vision/detection/detectionMode";

export interface WebDebugThresholds {
  yoloConf: number;
  yoloIou: number;
  boardConf: number;
  minBoardAreaRatio: number;
}

const DEFAULT_THRESHOLDS: WebDebugThresholds = {
  yoloConf: 0.3,
  yoloIou: 0.45,
  boardConf: 0.2,
  minBoardAreaRatio: 0.04,
};

interface WebDebugState {
  detectionMode: DetectionMode;
  thresholds: WebDebugThresholds;
  setDetectionMode: (mode: DetectionMode) => void;
  setThreshold: <K extends keyof WebDebugThresholds>(key: K, value: number) => void;
  resetThresholds: () => void;
}

export const useWebDebugStore = create<WebDebugState>((set) => ({
  detectionMode: getDetectionMode(),
  thresholds: { ...DEFAULT_THRESHOLDS },
  setDetectionMode: (detectionMode) => set({ detectionMode }),
  setThreshold: (key, value) =>
    set((s) => ({ thresholds: { ...s.thresholds, [key]: value } })),
  resetThresholds: () => set({ thresholds: { ...DEFAULT_THRESHOLDS } }),
}));
