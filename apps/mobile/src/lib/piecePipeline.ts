import { detectPieces } from "@/lib/api/detectPieces";
import { ApiError } from "@/lib/api/client";
import { PieceDetectionResult } from "@/types";

export type PieceProgressCallback = (step: "upload" | "localize" | "recognize") => void;

export async function runPieceDetectionPipeline(
  file: File,
  onProgress?: PieceProgressCallback,
): Promise<PieceDetectionResult> {
  const originalUrl = URL.createObjectURL(file);
  onProgress?.("upload");
  onProgress?.("localize");

  try {
    onProgress?.("recognize");
    return await detectPieces(file, originalUrl);
  } catch (err) {
    if (err instanceof ApiError) {
      throw new Error(err.message);
    }
    if (err instanceof TypeError) {
      const detail = err.message?.trim();
      throw new Error(
        detail && detail !== "Failed to fetch"
          ? `Error procesando la respuesta del servidor: ${detail}`
          : "No se pudo conectar con el servidor. ¿Está corriendo la API en el puerto 8001?",
      );
    }
    throw err;
  }
}
