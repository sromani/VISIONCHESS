export {
  WEB_DEBUG,
  MOBILE_PRODUCTION,
  DEV_MODE,
  IS_NATIVE,
} from "./flags";

export {
  API_BASE,
  VISION_LOCAL,
  USE_REMOTE_VISION,
  OFFLINE_MODE,
  LOCAL_GEOMETRY_FIRST,
  GEOMETRY_BACKEND,
  CAMERA_CORRECT_ORIENTATION,
  WARP_PREPROCESS,
  getDetectionMode,
  detectionModeLabel,
  DETECTION_MODES,
  type DetectionMode,
  type GeometryBackend,
} from "@/lib/config.runtime";
