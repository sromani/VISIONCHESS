import { base64ToDataUrl, apiFetch } from "@/lib/api/client";
import { WARP_PREPROCESS } from "@/lib/config";
import { BoardMatrixCell, DetectionDebug, DetectionResult, MlSquareDebug, SquareInfo } from "@/types";

interface DetectBoardApiResponse {
  job_id: string;
  corners: { x: number; y: number }[];
  confidence: number;
  original_width: number;
  original_height: number;
  output_width: number;
  output_height: number;
  warped_image_base64: string;
  fen: string | null;
  interactive_fen: string | null;
  fen_confidence: number | null;
  fen_valid: boolean | null;
  board_ready: boolean;
  board_matrix: { label: string; confidence: number }[][] | null;
  orientation: string | null;
  debug_overlay_base64: string | null;
  debug_montage_base64: string | null;
  debug: {
    original_base64?: string | null;
    detected_lines_base64?: string | null;
    intersections_base64?: string | null;
    mesh_base64?: string | null;
    rectified_board_base64?: string | null;
    rectified_upscaled_base64?: string | null;
    square_extraction_base64?: string | null;
    crop_quality_base64?: string | null;
    occupancy_base64?: string | null;
    occupancy_detail_base64?: string | null;
    classifier_confidence_base64?: string | null;
    ml_occupancy_base64?: string | null;
    ml_piece_top1_base64?: string | null;
    ml_onnx_occ_crops_base64?: string | null;
    ml_onnx_piece_crops_base64?: string | null;
    ml_detail_base64?: string | null;
    fen_candidates_base64?: string | null;
    mesh_quality_base64?: string | null;
    final_board_base64?: string | null;
    grid_debug_extreme_base64?: string | null;
    dataset_squares_base64?: string | null;
    rectified_grid_base64?: string | null;
    corners_original_base64?: string | null;
    corners_warped_base64?: string | null;
    rectified_preprocessed_base64?: string | null;
  } | null;
  squares: {
    name: string;
    filename: string;
    cell_bbox: number[];
    crop_bbox: number[];
    url: string;
    label?: string;
    confidence?: number;
    occupied?: boolean;
  }[];
  processing_ms: number;
  metadata: Record<string, unknown>;
}

function mapDebug(raw: DetectBoardApiResponse["debug"]): DetectionDebug | null {
  if (!raw) return null;
  const steps: DetectionDebug = {};
  const map: [keyof DetectionDebug, keyof NonNullable<typeof raw>][] = [
    ["original", "original_base64"],
    ["detectedLines", "detected_lines_base64"],
    ["intersections", "intersections_base64"],
    ["mesh", "mesh_base64"],
    ["rectifiedBoard", "rectified_board_base64"],
    ["rectifiedUpscaled", "rectified_upscaled_base64"],
    ["squareExtraction", "square_extraction_base64"],
    ["cropQuality", "crop_quality_base64"],
    ["occupancy", "occupancy_base64"],
    ["occupancyDetail", "occupancy_detail_base64"],
    ["classifierConfidence", "classifier_confidence_base64"],
    ["mlOccupancy", "ml_occupancy_base64"],
    ["mlPieceTop1", "ml_piece_top1_base64"],
    ["mlOnnxOccCrops", "ml_onnx_occ_crops_base64"],
    ["mlOnnxPieceCrops", "ml_onnx_piece_crops_base64"],
    ["mlDetail", "ml_detail_base64"],
    ["fenCandidates", "fen_candidates_base64"],
    ["meshQuality", "mesh_quality_base64"],
    ["finalBoard", "final_board_base64"],
    ["gridDebugExtreme", "grid_debug_extreme_base64"],
    ["rectifiedGrid", "rectified_grid_base64"],
    ["cornersOriginal", "corners_original_base64"],
    ["cornersWarped", "corners_warped_base64"],
    ["rectifiedPreprocessed", "rectified_preprocessed_base64"],
  ];
  for (const [target, source] of map) {
    const b64 = raw[source];
    if (b64) steps[target] = base64ToDataUrl(b64);
  }
  return Object.keys(steps).length > 0 ? steps : null;
}

function mapSquares(raw: DetectBoardApiResponse["squares"]): SquareInfo[] {
  return raw.map((sq) => ({
    name: sq.name,
    filename: sq.filename,
    cellBbox: sq.cell_bbox as SquareInfo["cellBbox"],
    cropBbox: sq.crop_bbox as SquareInfo["cropBbox"],
    label: sq.label,
    confidence: sq.confidence,
    occupied: sq.occupied,
  }));
}

function mapBoardMatrix(
  raw: DetectBoardApiResponse["board_matrix"],
): BoardMatrixCell[][] | undefined {
  if (!raw) return undefined;
  return raw.map((row) =>
    row.map((cell) => ({ label: cell.label, confidence: cell.confidence })),
  );
}

export async function detectBoard(
  file: File,
  originalUrl: string,
  endpoint = "/detect-board",
): Promise<DetectionResult> {
  const form = new FormData();
  form.append("file", file);

  const params = new URLSearchParams();
  if (endpoint.includes("lc2fen") && WARP_PREPROCESS && WARP_PREPROCESS !== "none") {
    params.set("warp_preprocess", WARP_PREPROCESS);
  }
  const qs = params.toString();
  const url = qs ? `${endpoint}?${qs}` : endpoint;

  const data = await apiFetch<DetectBoardApiResponse>(url, {
    method: "POST",
    body: form,
  });

  if (!data.fen && !data.squares?.length) {
    throw new Error("Classification pipeline did not return a FEN or square labels");
  }

  return {
    jobId: data.job_id,
    confidence: data.confidence,
    originalUrl,
    warpedUrl: base64ToDataUrl(data.warped_image_base64),
    overlayUrl: data.debug_overlay_base64
      ? base64ToDataUrl(data.debug_overlay_base64)
      : "/mock/overlay-board.svg",
    montageUrl: data.debug_montage_base64
      ? base64ToDataUrl(data.debug_montage_base64)
      : undefined,
    corners: data.corners,
    fen: data.fen ?? "",
    interactiveFen: data.interactive_fen,
    fenConfidence: data.fen_confidence ?? undefined,
    fenValid: data.fen_valid ?? undefined,
    boardReady: data.board_ready,
    boardMatrix: mapBoardMatrix(data.board_matrix),
    orientation: data.orientation ?? undefined,
    originalWidth: data.original_width,
    originalHeight: data.original_height,
    outputWidth: data.output_width,
    outputHeight: data.output_height,
    squares: mapSquares(data.squares),
    processingMs: data.processing_ms,
    debug: mapDebug(data.debug),
    mlDebug: mapMlDebug(data.metadata),
    metadata: data.metadata,
  };
}

function mapMlDebug(metadata: Record<string, unknown>): Record<string, MlSquareDebug> | null {
  const raw = metadata.ml_debug as { squares?: Record<string, MlSquareDebug> } | undefined;
  if (!raw?.squares) return null;
  return raw.squares;
}
