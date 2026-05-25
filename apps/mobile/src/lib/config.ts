/** Mobile app — debug panels hidden in production builds. */
export const DEV_MODE = import.meta.env.VITE_DEV_MODE === "true";

/** Optional remote API for dev only. Mobile runs offline-first when empty. */
export const API_BASE = import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "";

export const IS_NATIVE =
  typeof window !== "undefined" &&
  (window as Window & { Capacitor?: { isNativePlatform?: () => boolean } }).Capacitor
    ?.isNativePlatform?.() === true;

export const OFFLINE_MODE = !API_BASE;
