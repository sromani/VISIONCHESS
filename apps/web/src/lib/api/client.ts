export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      const raw = body.detail ?? body.message ?? detail;
      detail = Array.isArray(raw)
        ? raw.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ")
        : String(raw);
    } catch {
      /* ignore */
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export function base64ToDataUrl(base64: string, mime = "image/jpeg"): string {
  return `data:${mime};base64,${base64}`;
}
