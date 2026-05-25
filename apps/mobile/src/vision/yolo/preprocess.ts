import type { ImageRGBA } from "./types";

const PAD_VALUE = 114;

/**
 * YOLO letterbox preprocess — port of `_preprocess` in yolo_detector.py (RGB, pad 114).
 * Returns NCHW float32 blob [1, 3, inputSize, inputSize] and letterbox scale.
 */
export function preprocessYoloImage(
  image: ImageRGBA,
  inputSize: number,
): { blob: Float32Array; scale: number; dims: [number, number, number, number] } {
  const { width: w, height: h, data } = image;
  const scale = inputSize / Math.max(h, w);
  const newW = Math.round(w * scale);
  const newH = Math.round(h * scale);

  const rgb = new Float32Array(3 * inputSize * inputSize);
  rgb.fill(PAD_VALUE / 255);

  for (let y = 0; y < newH; y += 1) {
    for (let x = 0; x < newW; x += 1) {
      const srcX = Math.min(w - 1, Math.floor(x / scale));
      const srcY = Math.min(h - 1, Math.floor(y / scale));
      const srcIdx = (srcY * w + srcX) * 4;
      const r = data[srcIdx] / 255;
      const g = data[srcIdx + 1] / 255;
      const b = data[srcIdx + 2] / 255;
      const plane = inputSize * inputSize;
      const dst = y * inputSize + x;
      rgb[dst] = r;
      rgb[plane + dst] = g;
      rgb[2 * plane + dst] = b;
    }
  }

  return { blob: rgb, scale, dims: [1, 3, inputSize, inputSize] };
}
