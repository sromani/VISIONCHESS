import type { ImageRGBA } from "./yolo/types";

/** Fit image into a square canvas for YOLO when board warp is unavailable. */
export function resizeImageToSquare(image: ImageRGBA, size = 1200): ImageRGBA {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D not available");

  const src = document.createElement("canvas");
  src.width = image.width;
  src.height = image.height;
  const sctx = src.getContext("2d");
  if (!sctx) throw new Error("Canvas 2D not available");
  sctx.putImageData(
    new ImageData(new Uint8ClampedArray(image.data), image.width, image.height),
    0,
    0,
  );

  const scale = Math.min(size / image.width, size / image.height);
  const dw = Math.round(image.width * scale);
  const dh = Math.round(image.height * scale);
  const ox = Math.round((size - dw) / 2);
  const oy = Math.round((size - dh) / 2);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(src, ox, oy, dw, dh);

  const out = ctx.getImageData(0, 0, size, size);
  return { width: size, height: size, data: out.data };
}
