import type { YoloDetection } from "@/vision/yolo/types";
import type { ImageRGBA } from "@/vision/yolo/types";

/** Draw YOLO boxes on rectified board for WEB_DEBUG sidebar. */
export function drawYoloDebugImage(
  image: ImageRGBA,
  detections: YoloDetection[],
): string {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const pixels = new Uint8ClampedArray(image.data);
  ctx.putImageData(new ImageData(pixels, image.width, image.height), 0, 0);

  ctx.font = "12px monospace";
  for (const d of detections) {
    const { x, y, w, h } = d.bbox;
    const color = d.className === "board" ? "#00ff88" : "#ffcc00";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.fillText(`${d.className} ${(d.confidence * 100).toFixed(0)}%`, x + 2, y + 14);
  }

  return canvas.toDataURL("image/jpeg", 0.9);
}
