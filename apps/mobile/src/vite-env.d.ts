/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** @deprecated use VITE_WEB_DEBUG */
  readonly VITE_DEV_MODE?: string;
  /** Browser CV lab — never enabled on Capacitor native */
  readonly VITE_WEB_DEBUG?: string;
  readonly VITE_API_URL?: string;
  readonly VITE_VISION_LOCAL?: string;
  /** stable_v1 | yolo_v1 | mesh_v2 | experimental */
  readonly VITE_DETECTION_MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
