/**
 * Runtime flags — web CV lab vs Capacitor production.
 *
 * WEB_DEBUG: browser only, VITE_WEB_DEBUG=true, never on native.
 * MOBILE_PRODUCTION: native app or non-debug web build — no debug UI/overlays.
 */

export const IS_NATIVE =
  typeof window !== "undefined" &&
  (window as Window & { Capacitor?: { isNativePlatform?: () => boolean } }).Capacitor
    ?.isNativePlatform?.() === true;

/** CV lab in browser — overlays, benchmark, threshold sliders, pipeline inspector. */
export const WEB_DEBUG =
  !IS_NATIVE &&
  (import.meta.env.VITE_WEB_DEBUG === "true" ||
    import.meta.env.VITE_DEV_MODE === "true");

/** Clean product UI — iOS/Android Capacitor and web without VITE_WEB_DEBUG. */
export const MOBILE_PRODUCTION = IS_NATIVE || !WEB_DEBUG;

/** @deprecated Use WEB_DEBUG — kept so old .env entries do not break imports. */
export const DEV_MODE = WEB_DEBUG;
