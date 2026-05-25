import { MOCK_ANALYSIS, MOCK_DETECTION_META } from "@/mocks/data";
import { AnalysisResult, DetectionResult } from "@/types";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function mockDetectBoard(file: File): Promise<DetectionResult> {
  await delay(1200);
  const originalUrl = URL.createObjectURL(file);
  return {
    jobId: crypto.randomUUID(),
    originalUrl,
    warpedUrl: "/mock/warped-board.svg",
    overlayUrl: "/mock/overlay-board.svg",
    ...MOCK_DETECTION_META,
  };
}

export async function mockAnalyzePosition(_fen: string): Promise<AnalysisResult> {
  await delay(900);
  return MOCK_ANALYSIS;
}

export type ProgressCallback = (step: "upload" | "detect" | "split" | "analyze") => void;

export async function mockFullPipeline(
  file: File,
  onProgress?: ProgressCallback,
): Promise<{ detection: DetectionResult; analysis: AnalysisResult }> {
  onProgress?.("upload");
  await delay(400);

  onProgress?.("detect");
  const detection = await mockDetectBoard(file);

  onProgress?.("split");
  await delay(500);

  onProgress?.("analyze");
  const analysis = await mockAnalyzePosition(detection.fen);
  await delay(300);

  return { detection, analysis };
}
