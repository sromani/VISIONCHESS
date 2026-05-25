import { detectBoard } from "@/lib/api/detectBoard";
import { ApiError } from "@/lib/api/client";
import { DetectionResult } from "@/types";

export type ProgressCallback = (step: "upload" | "detect" | "classify" | "validate" | "analyze") => void;

export async function runVisionPipeline(
  file: File,
  onProgress?: ProgressCallback,
): Promise<{ detection: DetectionResult }> {
  const originalUrl = URL.createObjectURL(file);

  onProgress?.("upload");
  onProgress?.("detect");

  let detection: DetectionResult;
  try {
    detection = await detectLc2fen(file, originalUrl);
  } catch (err) {
    if (err instanceof ApiError) {
      throw new Error(err.message);
    }
    if (err instanceof TypeError) {
      throw new Error(
        "No se pudo conectar con el servidor de detección. ¿Está corriendo la API en el puerto 8001?",
      );
    }
    throw err;
  }

  onProgress?.("classify");
  onProgress?.("validate");
  onProgress?.("analyze");

  return { detection };
}

async function detectLc2fen(file: File, originalUrl: string): Promise<DetectionResult> {
  const form = new FormData();
  form.append("file", file);
  return detectBoard(file, originalUrl, "/detect-lc2fen");
}
