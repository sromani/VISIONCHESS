/** Mirrors ml/models/pretrained/yolov8_chess_pieces.json (bundled for offline YOLO). */
export const YOLO_MODEL_PATH = "/models/yolov8_chess_pieces.onnx";

export const YOLO_CLASS_NAMES = [
  "board",
  "white_king",
  "white_queen",
  "white_rook",
  "white_bishop",
  "white_knight",
  "white_pawn",
  "black_king",
  "black_queen",
  "black_rook",
  "black_bishop",
  "black_knight",
  "black_pawn",
] as const;

export const YOLO_CONFIG = {
  inputSize: 640,
  inputName: "images",
  confThreshold: 0.3,
  iouThreshold: 0.45,
  skipClasses: new Set(["board"]),
  /** Phase 1 debug: show fine-grained piece classes in UI. */
  localizationOnly: false,
  boardClassIndex: 0,
  coordsNormalized: true,
  maxBoxRatio: 0.32,
  minBoxPx: 18,
} as const;
