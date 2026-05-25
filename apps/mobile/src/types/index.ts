export type AppPhase = "idle" | "uploading" | "detecting" | "classifying" | "validating" | "analyzing" | "ready" | "error";

export interface Point {
  x: number;
  y: number;
}

export interface BoardMatrixCell {
  label: string;
  confidence: number;
}

export interface MlClassPrediction {
  label: string;
  probability: number;
  logit: number;
}

export interface MlSquareDebug {
  square_name: string;
  row: number;
  col: number;
  analysis_crop_shape: [number, number];
  occupancy?: {
    occupied_probability: number;
    empty_probability: number;
    logits: number[];
    model_input_size: number;
  };
  piece?: {
    top3: MlClassPrediction[];
    logits: number[];
    class_names: string[];
    model_input_size: number;
  };
}

export interface OccupancyModelInfo {
  source: string;
  path: string;
  image_size: number;
  num_classes: number;
  preprocess: string;
}

export interface PieceModelInfo {
  source: string;
  path: string;
  image_size: number;
  num_classes: number;
  class_names: string[];
  includes_empty_class: boolean;
  preprocess: string;
}

export interface TopPrediction {
  label: string;
  probability: number;
  logit: number;
}

export interface DetectorModelInfo {
  source: string;
  path: string;
  image_size: number;
  num_classes: number;
  class_names: string[];
  preprocess: string;
  model_type: string;
  conf_threshold: number;
  iou_threshold: number;
  localization_only?: boolean;
  role?: string;
}

export interface ClassifierModelInfo {
  source: string;
  path: string;
  image_size: number;
  num_classes: number;
  class_names: string[];
  preprocess: string;
  model_type: string;
  role?: string;
}

export interface PieceBBoxDetection {
  className: string;
  localizationConfidence: number;
  classifiedLabel?: string;
  classifiedConfidence?: number;
  bbox: [number, number, number, number];
  square?: string | null;
  top3: TopPrediction[];
  cropShape?: [number, number];
}

export interface ClassifierCropDebug {
  square: string;
  top3: TopPrediction[];
  cropUrl: string;
}

export interface SquareAssignment {
  squareName: string;
  row: number;
  col: number;
  occupied: boolean;
  label: string;
  confidence: number;
  pieceLabel: string;
  pieceConfidence: number;
  localizationConfidence: number;
  bbox?: [number, number, number, number] | null;
  top3: TopPrediction[];
  cellBbox: [number, number, number, number];
}

export interface PieceDetectionResult {
  jobId: string;
  mode: string;
  confidence: number;
  originalUrl: string;
  originalWidth: number;
  originalHeight: number;
  outputWidth: number;
  outputHeight: number;
  rectifiedBoardUrl: string;
  localizationOverlayUrl?: string;
  classifierOverlayUrl?: string;
  classifierCropsUrl?: string;
  detector: DetectorModelInfo;
  classifier: ClassifierModelInfo;
  detections: PieceBBoxDetection[];
  squares: SquareAssignment[];
  classifierCrops: ClassifierCropDebug[];
  occupiedCount: number;
  emptyCount: number;
  processingMs: number;
  metadata: Record<string, unknown>;
  debug?: {
    original?: string;
    rectifiedUpscaled?: string;
  };
}

/** @deprecated legacy square classifier — full scan pipeline only */
export interface SquarePieceDetection {
  squareName: string;
  row: number;
  col: number;
  occupied: boolean;
  occupancyProbability: number;
  emptyProbability: number;
  pieceLabel: string;
  pieceConfidence: number;
  fusedConfidence: number;
  fusionMode: string;
  fusedEmptyThreshold: number;
  hardGateOccupied: boolean;
  hardGateLabel: string;
  occupancyThreshold: number;
  pieceClassifierRan: boolean;
  label: string;
  confidence: number;
  top3: MlClassPrediction[];
  logits: number[];
  classNames: string[];
  cellBbox: [number, number, number, number];
  contextPieceLabel?: string | null;
  contextPieceConfidence?: number | null;
  contextTop3?: MlClassPrediction[];
}

export interface FusionExperiment {
  hard_gate: { occupied_count: number; empty_count: number; threshold: number };
  soft_fusion: { occupied_count: number; empty_count: number; alpha: number; beta: number };
  delta: { pieces_recovered_by_soft: string[]; pieces_lost_by_soft: string[]; recovered_count: number; lost_count: number };
}

export interface CropExperiment {
  tight: { strategy: string; squares: number };
  context: { strategy: string; squares: number };
  context_scale?: number;
  agreement_count: number;
  disagreement_count: number;
  context_higher_confidence: string[];
  tight_higher_confidence: string[];
  large_piece_disagreements: {
    square_name: string;
    tight_label: string;
    tight_confidence: number;
    context_label: string;
    context_confidence: number;
  }[];
}

export interface DetectionDebug {
  original?: string;
  detectedLines?: string;
  intersections?: string;
  mesh?: string;
  rectifiedBoard?: string;
  rectifiedUpscaled?: string;
  squareExtraction?: string;
  cropQuality?: string;
  occupancy?: string;
  occupancyDetail?: string;
  classifierConfidence?: string;
  mlOccupancy?: string;
  mlPieceTop1?: string;
  mlOnnxOccCrops?: string;
  mlOnnxPieceCrops?: string;
  mlDetail?: string;
  fenCandidates?: string;
  meshQuality?: string;
  finalBoard?: string;
  classificationGrid?: string;
  gridDebugExtreme?: string;
}

export interface SquareInfo {
  name: string;
  filename: string;
  cellBbox: [number, number, number, number];
  cropBbox: [number, number, number, number];
  label?: string;
  confidence?: number;
  occupied?: boolean;
}

export interface DetectionResult {
  jobId: string;
  confidence: number;
  originalUrl: string;
  warpedUrl: string;
  overlayUrl: string;
  montageUrl?: string;
  corners: Point[];
  /** Best-effort FEN from pipeline — may fail validation */
  fen: string;
  /** Legal FEN for interactive board — only set when boardReady */
  interactiveFen: string | null;
  fenConfidence?: number;
  fenValid?: boolean;
  boardReady: boolean;
  boardMatrix?: BoardMatrixCell[][];
  orientation?: string;
  originalWidth: number;
  originalHeight: number;
  outputWidth: number;
  outputHeight: number;
  squares: SquareInfo[];
  processingMs: number;
  debug?: DetectionDebug | null;
  mlDebug?: Record<string, MlSquareDebug> | null;
  metadata?: Record<string, unknown>;
}

export interface EngineLine {
  multipv: number;
  move: string;
  from: string;
  to: string;
  san?: string;
  evalCp: number | null;
  evalMate: number | null;
  pv: string[];
}

export interface AnalysisResult {
  evaluationCp: number | null;
  evaluationMate: number | null;
  bestMove: string;
  bestMoveSan?: string;
  depth: number;
  nodesSearched?: number;
  nps?: number;
  lines: EngineLine[];
  processingMs: number;
}

export type PipelineStep = "upload" | "detect" | "classify" | "validate" | "analyze";

export interface PipelineStepState {
  id: PipelineStep;
  label: string;
  status: "pending" | "active" | "done";
}
