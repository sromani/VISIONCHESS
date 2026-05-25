#!/usr/bin/env node
/**
 * Copies ONNX + ORT wasm into public/ for on-device YOLO.
 */
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const publicDir = path.join(root, "public");
const modelsDir = path.join(publicDir, "models");
const ortDir = path.join(publicDir, "ort");

const repoRoot = path.join(root, "..", "..");
const yoloSrc = path.join(repoRoot, "ml", "models", "pretrained", "yolov8_chess_pieces.onnx");
const yoloJsonSrc = path.join(repoRoot, "ml", "models", "pretrained", "yolov8_chess_pieces.json");
const yoloDest = path.join(modelsDir, "yolov8_chess_pieces.onnx");
const yoloJsonDest = path.join(modelsDir, "yolov8_chess_pieces.json");
const lapsSrc = path.join(
  repoRoot,
  "ml",
  "vendor",
  "LiveChess2FEN",
  "lc2fen",
  "detectboard",
  "models",
  "laps_model.onnx",
);
const lapsDest = path.join(modelsDir, "laps_model.onnx");

const ortPkg = path.join(root, "node_modules", "onnxruntime-web", "dist");

function copyFile(src, dest) {
  if (!fs.existsSync(src)) {
    console.warn(`Skip (missing): ${src}`);
    return false;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
  const mb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(1);
  console.log(`Copied ${path.basename(dest)} (${mb} MB)`);
  return true;
}

fs.mkdirSync(modelsDir, { recursive: true });
fs.mkdirSync(ortDir, { recursive: true });

const hasYolo = copyFile(yoloSrc, yoloDest);
copyFile(yoloJsonSrc, yoloJsonDest);
const hasLaps = copyFile(lapsSrc, lapsDest);

if (!hasYolo) {
  console.warn(
    "\nYOLO ONNX not found. Download with:\n  python ml/scripts/setup_yolo_chess.py --model nakst\n",
  );
}
if (!hasLaps) {
  console.warn(
    "\nLAPS ONNX not found. Download with:\n  python ml/scripts/setup_lc2fen.py\n",
  );
}

// onnxruntime-web 1.22 defaults to the JSEP threaded backend; include both variants.
const ortFiles = [
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort.wasm.min.mjs",
];

for (const name of ortFiles) {
  copyFile(path.join(ortPkg, name), path.join(ortDir, name));
}

console.log("Vision assets copy done.");
