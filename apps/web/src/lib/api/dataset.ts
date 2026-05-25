import { apiFetch } from "./client";

export type DatasetSample = {
  label: string;
  path: string;
};

export type DatasetSamplesResponse = {
  job_id: string;
  samples: DatasetSample[];
};

export async function fetchDatasetSamples(jobId: string): Promise<DatasetSamplesResponse> {
  return apiFetch(`/dataset/${jobId}/samples`);
}

export async function relabelSample(
  jobId: string,
  samplePath: string,
  newLabel: string,
): Promise<{ ok: boolean; new_path: string }> {
  return apiFetch(`/dataset/${jobId}/relabel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_path: samplePath, new_label: newLabel }),
  });
}

export const PIECE_LABELS = [
  "empty",
  "white_pawn",
  "white_knight",
  "white_bishop",
  "white_rook",
  "white_queen",
  "white_king",
  "black_pawn",
  "black_knight",
  "black_bishop",
  "black_rook",
  "black_queen",
  "black_king",
] as const;
