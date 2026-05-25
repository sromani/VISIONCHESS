const MAX_EDGE = 160;
const JPEG_QUALITY = 0.72;

export async function createImagePreview(file: File): Promise<string | undefined> {
  if (!file.type.startsWith("image/")) return undefined;

  try {
    const bitmap = await createImageBitmap(file);
    const scale = MAX_EDGE / Math.max(bitmap.width, bitmap.height, 1);
    const width = Math.max(1, Math.round(bitmap.width * Math.min(1, scale)));
    const height = Math.max(1, Math.round(bitmap.height * Math.min(1, scale)));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return undefined;

    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
  } catch {
    return undefined;
  }
}
