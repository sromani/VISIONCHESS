import { Camera, CameraResultType, CameraSource } from "@capacitor/camera";
import { Capacitor } from "@capacitor/core";

function isUserCancel(err: unknown): boolean {
  const msg =
    err && typeof err === "object" && "message" in err
      ? String((err as { message: string }).message)
      : String(err);
  return /cancel/i.test(msg);
}

async function getPhotoSafe(
  options: Parameters<typeof Camera.getPhoto>[0],
): Promise<Awaited<ReturnType<typeof Camera.getPhoto>> | null> {
  try {
    return await Camera.getPhoto(options);
  } catch (err) {
    if (isUserCancel(err)) return null;
    throw err;
  }
}

function dataUrlToFile(dataUrl: string, fileName: string): File {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)?.[1] ?? "image/jpeg";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new File([bytes], fileName, { type: mime });
}

async function pickFromWebInput(accept: string): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.onchange = () => {
      resolve(input.files?.[0] ?? null);
    };
    input.click();
  });
}

export async function captureBoardPhoto(): Promise<File | null> {
  if (!Capacitor.isNativePlatform()) {
    return pickFromWebInput("image/png,image/jpeg,image/webp");
  }

  const photo = await getPhotoSafe({
    quality: 92,
    allowEditing: false,
    resultType: CameraResultType.DataUrl,
    source: CameraSource.Camera,
    correctOrientation: true,
    saveToGallery: false,
  });
  if (!photo) return null;

  if (!photo.dataUrl) return null;
  return dataUrlToFile(photo.dataUrl, `board-${Date.now()}.jpg`);
}

export async function pickBoardFromGallery(): Promise<File | null> {
  if (!Capacitor.isNativePlatform()) {
    return pickFromWebInput("image/png,image/jpeg,image/webp");
  }

  const photo = await getPhotoSafe({
    quality: 92,
    allowEditing: false,
    resultType: CameraResultType.DataUrl,
    source: CameraSource.Photos,
    correctOrientation: true,
  });
  if (!photo) return null;

  if (!photo.dataUrl) return null;
  return dataUrlToFile(photo.dataUrl, `board-${Date.now()}.jpg`);
}
