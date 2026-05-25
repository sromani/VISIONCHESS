import { Capacitor } from "@capacitor/core";
import { Filesystem, Directory } from "@capacitor/filesystem";

import { geoDebugLog } from "@/lib/vision/geometryDebugLogger";
import type { OfflineGeometryResult } from "@/lib/vision/offlineBoardGeometry";
import type { NativeOfflineRecognizeResult } from "@/lib/vision/offlineYolo/types";
import type { DetectionResult } from "@/types";

const OFFLINE_DEBUG = import.meta.env.VITE_OFFLINE_DEBUG === "true";

function dataUrlToBase64(dataUrl: string): string {
  const i = dataUrl.indexOf(",");
  return i >= 0 ? dataUrl.slice(i + 1) : dataUrl;
}

async function writeBase64(path: string, dataUrlOrB64: string): Promise<void> {
  const data = dataUrlOrB64.includes(",") ? dataUrlToBase64(dataUrlOrB64) : dataUrlOrB64;
  await Filesystem.writeFile({
    path,
    data,
    directory: Directory.Data,
  });
}

export async function saveOfflineDebugBundle(
  geo: OfflineGeometryResult,
  native: NativeOfflineRecognizeResult,
  detection: DetectionResult,
): Promise<void> {
  if (!OFFLINE_DEBUG || !Capacitor.isNativePlatform()) return;

  const jobId = detection.jobId.replace(/[^a-zA-Z0-9_-]/g, "_");
  const base = `offline-debug/${jobId}`;

  try {
    const warped = geo.debug?.rectifiedBoard ?? geo.warpedUrl;
    if (warped) await writeBase64(`${base}/01_warped.jpg`, warped);
    if (native.debug.overlayJpegBase64) {
      await writeBase64(`${base}/02_yolo_overlay.jpg`, native.debug.overlayJpegBase64);
    }

    const manifest = {
      savedAt: new Date().toISOString(),
      placementFen: native.placementFen,
      fen: detection.fen,
      boardReady: detection.boardReady,
      timings: native.timings,
      metadata: detection.metadata,
      detections: native.detections,
      logs: native.debug.logLines,
    };

    await Filesystem.writeFile({
      path: `${base}/manifest.json`,
      data: btoa(unescape(encodeURIComponent(JSON.stringify(manifest, null, 2)))),
      directory: Directory.Data,
    });

    geoDebugLog("offlineDebugStore.ts", "saved_bundle", { base }, "H-native");
  } catch (err) {
    geoDebugLog(
      "offlineDebugStore.ts",
      "save_failed",
      { error: err instanceof Error ? err.message : String(err) },
      "H-native",
    );
  }
}
