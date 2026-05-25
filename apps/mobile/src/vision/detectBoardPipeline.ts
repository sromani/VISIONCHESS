import type { DetectionResult } from "@/types";
import { drawYoloDebugImage } from "@/debug/web/drawYoloDebugImage";
import { useWebDebugStore } from "@/debug/web/webDebugStore";
import { WEB_DEBUG } from "@/shared/config/flags";

import { getDetectionMode, type DetectionMode } from "./detection/detectionMode";
import { captureFailureCase } from "./debug/failureCapture";
import { boardRegionFromDetections } from "./yolo/boardRegion";
import {
  detectBoardRegionInImage,
  fileToImageData,
  runBoardScanRaw,
  runYoloOnImage,
} from "./visionClient";
import {
  drawCornerOverlay,
  drawLocalizationDebugOverlay,
  imageDataToObjectUrl,
  rectifyBoardImage,
} from "./geometryClient";
import { resizeImageToSquare } from "./imageResize";
import { localYoloToDetection } from "./localYoloToDetection";
import type { LocalizationStatus, RectifyResult } from "./geometry/types";
import type { ImageRGBA } from "./yolo/types";

const MODEL_HINT =
  "Modelos no encontrados. Ejecuta:\n  python ml/scripts/setup_yolo_chess.py --model nakst\n  python ml/scripts/setup_lc2fen.py\n  npm run copy:vision-assets --prefix apps/mobile";

export interface DetectBoardOptions {
  file: File;
  mode?: DetectionMode;
}

function attachMetadata(
  detection: DetectionResult,
  extras: Record<string, unknown>,
): DetectionResult {
  detection.metadata = { ...detection.metadata, ...extras };
  return detection;
}

async function runGeometry(
  originalImage: ImageRGBA,
  boardRegion: Awaited<ReturnType<typeof detectBoardRegionInImage>>,
  mode: DetectionMode,
): Promise<RectifyResult> {
  return rectifyBoardImage(originalImage, {
    corners: boardRegion?.corners,
    yoloConfidence: boardRegion?.confidence,
    detectionMode: mode,
  });
}

function resolveYoloTarget(
  geometry: RectifyResult,
  originalImage: ImageRGBA,
): { image: ImageRGBA; previewUrl: string } {
  if (geometry.geometryOk && geometry.corners.length === 4) {
    return {
      image: geometry.rectified,
      previewUrl: imageDataToObjectUrl(geometry.rectified),
    };
  }
  const square = resizeImageToSquare(originalImage, 1200);
  return { image: square, previewUrl: imageDataToObjectUrl(square) };
}

function buildOverlay(geometry: RectifyResult, originalImage: ImageRGBA): string {
  if (geometry.corners.length !== 4) return "";
  if (WEB_DEBUG && geometry.localizationDebug?.candidates.length) {
    return drawLocalizationDebugOverlay(
      originalImage,
      geometry.localizationDebug,
      geometry.corners,
    );
  }
  return drawCornerOverlay(originalImage, geometry.corners);
}

function userMessage(
  boardFound: boolean,
  status: LocalizationStatus,
  pieceCount: number,
  boardReady: boolean,
): string | undefined {
  if (!boardFound && pieceCount < 2) {
    return "No se encontró tablero en la foto. Probá acercar el tablero o mejorar la luz.";
  }
  if (status === "weak" && !boardReady) {
    return "Tablero detectado con baja confianza. Revisá la posición en Setup o escaneá de nuevo.";
  }
  return undefined;
}

/**
 * Unified offline detect — mode stable_v1 by default.
 */
export async function detectBoardFromFile(options: DetectBoardOptions): Promise<DetectionResult> {
  const mode =
    options.mode ??
    (WEB_DEBUG ? useWebDebugStore.getState().detectionMode : undefined) ??
    getDetectionMode();

  try {
    const original = await fileToImageData(options.file, 1920);
    const originalImage: ImageRGBA = {
      width: original.width,
      height: original.height,
      data: original.data,
    };

    let boardScanRaw: Awaited<ReturnType<typeof runBoardScanRaw>> | null = null;
    let boardRegion: Awaited<ReturnType<typeof detectBoardRegionInImage>>;
    if (WEB_DEBUG) {
      boardScanRaw = await runBoardScanRaw(originalImage);
      boardRegion = boardRegionFromDetections(
        boardScanRaw.detections,
        originalImage.width,
        originalImage.height,
      );
    } else {
      boardRegion = await detectBoardRegionInImage(originalImage);
    }
    const geometry = await runGeometry(originalImage, boardRegion, mode);

    if (boardRegion && geometry.metrics.method === "yolo_board_bbox") {
      geometry.metrics.boardConfidence = boardRegion.confidence;
    }

    const status: LocalizationStatus =
      geometry.localizationStatus ??
      (geometry.corners.length === 4 ? "weak" : "not_found");

    const boardFound =
      geometry.boardFound ?? (geometry.corners.length === 4 && status !== "not_found");

    const { image: yoloImage, previewUrl } = resolveYoloTarget(geometry, originalImage);
    const local = await runYoloOnImage(yoloImage, previewUrl);

    const pieceCount = local.detections.filter(
      (d) => d.className !== "board" && d.confidence >= 0.2,
    ).length;

    const overlayUrl = buildOverlay(geometry, originalImage) || original.imageUrl;

    const detection = localYoloToDetection(local, {
      originalUrl: original.imageUrl,
      originalWidth: original.width,
      originalHeight: original.height,
      originalImage,
      geometry: {
        ...geometry,
        boardFound,
        localizationStatus: status,
        geometryOk: geometry.corners.length === 4,
      },
      overlayUrl,
    });

    attachMetadata(detection, {
      detection_mode: mode,
      geometryMs: geometry.metrics.geometryMs,
      geometryMethod: geometry.metrics.method,
      localization_status: status,
      board_found: boardFound,
      localization_score: geometry.metrics.localizationScore ?? 0,
      localization_attempts: geometry.localizationDebug?.attempts,
      localization_best_source: geometry.localizationDebug?.bestSource,
      localization_rejection: geometry.localizationDebug?.rejectionReason,
      yolo_piece_boxes: pieceCount,
      board_localization: geometry.localizationDebug,
      yolo_board_detected: Boolean(boardRegion),
      yolo_board_scan: boardScanRaw?.detections.map((d) => ({
        label: d.className,
        confidence: d.confidence,
        bbox: [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
      })),
      web_debug: WEB_DEBUG,
    });

    if (WEB_DEBUG) {
      detection.debug = {
        ...detection.debug,
        mlPieceTop1: drawYoloDebugImage(yoloImage, local.detections),
      };
    }

    const msg = userMessage(boardFound, status, pieceCount, detection.boardReady);
    if (msg) {
      detection.metadata!.user_message = msg;
    }

    const failed =
      !boardFound ||
      status === "not_found" ||
      (!detection.boardReady && pieceCount < 4);

    if (WEB_DEBUG && failed) {
      captureFailureCase({
        mode,
        reason: geometry.localizationDebug?.rejectionReason ?? msg ?? "detection_failed",
        fileName: options.file.name,
        geometry,
        detection,
        originalImage,
        overlayDataUrl: overlayUrl,
      });
    }

    return detection;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (
      msg.includes("ort") ||
      msg.includes("wasm") ||
      msg.includes("onnxruntime") ||
      msg.includes("no available backend")
    ) {
      throw new Error(
        `ONNX runtime no pudo cargar en este dispositivo. Rebuild nativo:\n  cd apps/mobile && npm run build && npm run cap:sync\n\nSi persiste, copiá ort:\n  npm run copy:vision-assets\n\n(${msg})`,
      );
    }
    if (
      msg.includes("YOLO model") ||
      msg.includes("LAPS model") ||
      msg.includes("404") ||
      msg.includes("not found")
    ) {
      throw new Error(`${MODEL_HINT}\n\n(${msg})`);
    }
    throw err instanceof Error ? err : new Error(msg);
  }
}
