/**
 * Public assets (public/ort, public/models) are always served from the site root,
 * NOT relative to the worker script URL (Vite dev workers live under /src/vision/).
 */
export function workerAssetUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  const base =
    typeof import.meta !== "undefined" &&
    import.meta.env &&
    typeof import.meta.env.BASE_URL === "string"
      ? import.meta.env.BASE_URL
      : "/";
  const root = base.startsWith("http")
    ? base
    : new URL(base, self.location.origin).href;
  return new URL(normalized.slice(1), root.endsWith("/") ? root : `${root}/`).href;
}

export async function fetchArrayBuffer(url: string, label: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${label} not found (${res.status}): ${url}`);
  }
  return res.arrayBuffer();
}
