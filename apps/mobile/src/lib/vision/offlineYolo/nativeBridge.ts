import { Capacitor, registerPlugin } from "@capacitor/core";

import { OFFLINE_MODE } from "@/lib/config";
import { geoDebugLog } from "@/lib/vision/geometryDebugLogger";
import { saveOfflineDebugBundle } from "@/lib/vision/offlineYolo/offlineDebugStore";
import type { OfflineGeometryResult } from "@/lib/vision/offlineBoardGeometry";
import { mapNativeYoloToDetection } from "@/lib/vision/offlineYolo/mapNativeResult";
import type { NativeOfflineRecognizeResult } from "@/lib/vision/offlineYolo/types";
import type { DetectionResult } from "@/types";

interface VisionChessOfflinePlugin {
  recognizeFromWarpedJpeg(options: {
    jpegBase64: string;
    width: number;
    height: number;
    confThreshold?: number;
  }): Promise<NativeOfflineRecognizeResult>;
}

const VisionChessOffline = registerPlugin<VisionChessOfflinePlugin>("VisionChessOffline", {
  web: () =>
    import("@/lib/vision/offlineYolo/webStub").then((m) => ({
      default: m.VisionChessOfflineWeb,
    })),
});

/** M1: on iOS with empty API URL, native YOLO is enabled unless explicitly disabled. */
export const OFFLINE_NATIVE_YOLO =
  Capacitor.getPlatform() === "ios" &&
  OFFLINE_MODE &&
  import.meta.env.VITE_OFFLINE_NATIVE_YOLO !== "false";

function dataUrlToRawBase64(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

export function canRunNativeOfflineYolo(): boolean {
  return OFFLINE_NATIVE_YOLO && Capacitor.isNativePlatform();
}

export async function recognizePiecesNative(
  geo: OfflineGeometryResult,
  originalUrl: string,
): Promise<DetectionResult | null> {
  if (!canRunNativeOfflineYolo()) {
    return null;
  }

  const warped = geo.debug?.rectifiedBoard ?? geo.warpedUrl;
  if (!warped) {
    geoDebugLog("nativeBridge.ts", "skip_no_warped", {}, "H-native");
    return null;
  }

  const started = performance.now();
  try {
    const raw = await VisionChessOffline.recognizeFromWarpedJpeg({
      jpegBase64: dataUrlToRawBase64(warped),
      width: geo.outputWidth,
      height: geo.outputHeight,
      confThreshold: 0.3,
    });
    const result = raw as NativeOfflineRecognizeResult;

    geoDebugLog(
      "nativeBridge.ts",
      "native_yolo_ok",
      {
        ms: Math.round(performance.now() - started),
        inferMs: result.timings.inferenceMs,
        detections: result.detections.length,
        placement: result.placementFen?.slice(0, 40),
      },
      "H-native",
    );

    const detection = mapNativeYoloToDetection(geo, result, originalUrl);
    await saveOfflineDebugBundle(geo, result, detection);
    return detection;
  } catch (err) {
    geoDebugLog(
      "nativeBridge.ts",
      "native_yolo_fail",
      { error: err instanceof Error ? err.message : String(err) },
      "H-native",
    );
    return null;
  }
}
