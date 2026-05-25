/**
 * Copy ONNX assets into the iOS Capacitor plugin Resources folder.
 * Run from repo root: node apps/mobile/scripts/copy-offline-models.cjs
 */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..", "..", "..");
const yoloSrc = path.join(root, "ml", "models", "pretrained", "yolov8_chess_pieces.onnx");
const lapsSrc = path.join(
  root,
  "ml",
  "vendor",
  "LiveChess2FEN",
  "lc2fen",
  "detectboard",
  "models",
  "laps_model.onnx",
);
const destDir = path.join(root, "packages", "visionchess-offline", "ios", "Resources");

function copy(src, name) {
  if (!fs.existsSync(src)) {
    console.warn(`[skip] missing: ${src}`);
    return false;
  }
  fs.mkdirSync(destDir, { recursive: true });
  const dest = path.join(destDir, name);
  fs.copyFileSync(src, dest);
  const mb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(1);
  console.log(`[ok] ${name} (${mb} MB) -> ${dest}`);
  return true;
}

console.log("Copying offline ONNX models for iOS bundle…");
const yolo = copy(yoloSrc, "yolov8_chess_pieces.onnx");
const laps = copy(lapsSrc, "laps_model.onnx");
if (!yolo) {
  console.error("Run: python ml/scripts/setup_yolo_chess.py");
  process.exit(1);
}
if (!laps) {
  console.warn("LAPS model missing — Phase 2 only. Run: python ml/scripts/setup_lc2fen.py");
}
