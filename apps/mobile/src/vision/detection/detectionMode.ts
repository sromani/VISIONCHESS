/**
 * Board detection pipeline versions — switch via VITE_DETECTION_MODE in .env
 */
export const DETECTION_MODES = [
  "stable_v1",
  "yolo_v1",
  "mesh_v2",
  "experimental",
] as const;

export type DetectionMode = (typeof DETECTION_MODES)[number];

const DEFAULT_MODE: DetectionMode = "stable_v1";

export function getDetectionMode(): DetectionMode {
  const raw = import.meta.env.VITE_DETECTION_MODE as string | undefined;
  if (raw && (DETECTION_MODES as readonly string[]).includes(raw)) {
    return raw as DetectionMode;
  }
  return DEFAULT_MODE;
}

export function detectionModeLabel(mode: DetectionMode): string {
  switch (mode) {
    case "stable_v1":
      return "Stable v1 (YOLO hint → warp, else contour+LAPS)";
    case "yolo_v1":
      return "YOLO board bbox only";
    case "mesh_v2":
      return "Mesh v2 (not implemented — falls back to stable_v1)";
    case "experimental":
      return "Experimental (multi-candidate search)";
    default:
      return mode;
  }
}
