import { WebPlugin } from "@capacitor/core";

import type { NativeOfflineRecognizeResult } from "@/lib/vision/offlineYolo/types";

/** Browser dev: native YOLO not available. */
export class VisionChessOfflineWeb extends WebPlugin {
  async recognizeFromWarpedJpeg(): Promise<NativeOfflineRecognizeResult> {
    throw new Error(
      "VisionChessOffline.recognizeFromWarpedJpeg requires an iOS build (npx cap run ios).",
    );
  }
}
