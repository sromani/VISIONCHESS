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

/**
 * When API is set: default true = fast local warp, then /detect-lc2fen for YOLO pieces.
 * Set VITE_LOCAL_GEOMETRY_FIRST=false for a single server-side /detect-lc2fen call.
 */
export const LOCAL_GEOMETRY_FIRST =
  !API_BASE || import.meta.env.VITE_LOCAL_GEOMETRY_FIRST !== "false";

/** Offline board warp: fast = Canvas/JS. opencv = full WASM (slow on phone). */
export type GeometryBackend = "fast" | "opencv";
export const GEOMETRY_BACKEND: GeometryBackend =
  import.meta.env.VITE_GEOMETRY_BACKEND === "opencv" ? "opencv" : "fast";

export const CAMERA_CORRECT_ORIENTATION =
  import.meta.env.VITE_CAMERA_CORRECT_ORIENTATION !== "false";

export const WARP_PREPROCESS =
  (import.meta.env.VITE_WARP_PREPROCESS as string | undefined)?.trim() || "none";
