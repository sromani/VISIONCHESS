import { base64ToDataUrl, apiFetch } from "@/lib/api/client";
import {
  ClassifierModelInfo,
  DetectorModelInfo,
  PieceBBoxDetection,
  PieceDetectionResult,
  SquareAssignment,
  TopPrediction,
} from "@/types";

interface PieceDetectionApiResponse {
  job_id: string;
  mode: string;
  confidence: number;
  original_width: number;
  original_height: number;
  output_width: number;
  output_height: number;
  rectified_board_base64: string;
  detector: DetectorModelInfo;
  classifier: ClassifierModelInfo;
  detections: {
    class: string;
    localization_confidence: number;
    classified_label?: string | null;
    classified_confidence?: number | null;
    bbox: number[];
    square?: string | null;
    top3?: TopPrediction[];
    crop_shape?: number[] | null;
  }[];
  squares: {
    square_name: string;
    row: number;
    col: number;
    occupied: boolean;
    label: string;
    confidence: number;
    piece_label: string;
    piece_confidence: number;
    localization_confidence: number;
    bbox?: number[] | null;
    top3?: TopPrediction[];
    cell_bbox: number[];
  }[];
  classifier_crops?: {
    square: string;
    top3?: TopPrediction[];
    crop_base64: string;
  }[];
  occupied_count: number;
  empty_count: number;
  debug: {
    original_base64?: string | null;
    rectified_upscaled_base64?: string | null;
    yolo_localization_base64?: string | null;
    yolo_overlay_base64?: string | null;
    classifier_overlay_base64?: string | null;
    piece_overlay_base64?: string | null;
    classifier_crops_base64?: string | null;
  } | null;
  processing_ms: number;
  metadata: Record<string, unknown>;
}

export async function detectPieces(file: File, originalUrl: string): Promise<PieceDetectionResult> {
  const form = new FormData();
  form.append("file", file);

  const data = await apiFetch<PieceDetectionApiResponse>("/detect-pieces", {
    method: "POST",
    body: form,
  });

  const debug = data.debug;
  return {
    jobId: data.job_id,
    mode: data.mode,
    confidence: data.confidence,
    originalUrl,
    originalWidth: data.original_width,
    originalHeight: data.original_height,
    outputWidth: data.output_width,
    outputHeight: data.output_height,
    rectifiedBoardUrl: base64ToDataUrl(data.rectified_board_base64),
    localizationOverlayUrl: debug?.yolo_localization_base64
      ? base64ToDataUrl(debug.yolo_localization_base64)
      : debug?.yolo_overlay_base64
        ? base64ToDataUrl(debug.yolo_overlay_base64)
        : undefined,
    classifierOverlayUrl: debug?.classifier_overlay_base64
      ? base64ToDataUrl(debug.classifier_overlay_base64)
      : debug?.piece_overlay_base64
        ? base64ToDataUrl(debug.piece_overlay_base64)
        : undefined,
    classifierCropsUrl: debug?.classifier_crops_base64
      ? base64ToDataUrl(debug.classifier_crops_base64)
      : undefined,
    detector: data.detector,
    classifier: data.classifier,
    detections: data.detections.map(mapDetection),
    squares: data.squares.map(mapSquare),
    classifierCrops: (data.classifier_crops ?? []).map((c) => ({
      square: c.square,
      top3: c.top3 ?? [],
      cropUrl: c.crop_base64 ? base64ToDataUrl(c.crop_base64, "image/jpeg") : "",
    })),
    occupiedCount: data.occupied_count,
    emptyCount: data.empty_count,
    processingMs: data.processing_ms,
    metadata: data.metadata,
    debug: {
      original: debug?.original_base64 ? base64ToDataUrl(debug.original_base64) : undefined,
      rectifiedUpscaled: debug?.rectified_upscaled_base64
        ? base64ToDataUrl(debug.rectified_upscaled_base64)
        : undefined,
    },
  };
}

function mapDetection(d: PieceDetectionApiResponse["detections"][0]): PieceBBoxDetection {
  return {
    className: d.class,
    localizationConfidence: d.localization_confidence,
    classifiedLabel: d.classified_label ?? undefined,
    classifiedConfidence: d.classified_confidence ?? undefined,
    bbox: d.bbox as [number, number, number, number],
    square: d.square ?? undefined,
    top3: d.top3 ?? [],
    cropShape: d.crop_shape as [number, number] | undefined,
  };
}

function mapSquare(s: PieceDetectionApiResponse["squares"][0]): SquareAssignment {
  return {
    squareName: s.square_name,
    row: s.row,
    col: s.col,
    occupied: s.occupied,
    label: s.label,
    confidence: s.confidence,
    pieceLabel: s.piece_label,
    pieceConfidence: s.piece_confidence,
    localizationConfidence: s.localization_confidence,
    bbox: s.bbox ? (s.bbox as [number, number, number, number]) : undefined,
    top3: s.top3 ?? [],
    cellBbox: s.cell_bbox as [number, number, number, number],
  };
}
