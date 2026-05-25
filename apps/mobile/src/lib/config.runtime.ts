/** Runtime config without debug flags (imported by shared/config). */

export const API_BASE = import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "";

export const VISION_LOCAL = import.meta.env.VITE_VISION_LOCAL === "true";

export {
  getDetectionMode,
  detectionModeLabel,
  DETECTION_MODES,
  type DetectionMode,
} from "@/vision/detection/detectionMode";

export const USE_REMOTE_VISION = Boolean(API_BASE) && !VISION_LOCAL;

export const OFFLINE_MODE = !API_BASE;
