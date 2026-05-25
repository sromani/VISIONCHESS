"use client";

import { useCallback, useEffect, useState } from "react";

import { Card } from "@/components/ui/Card";
import { Spinner } from "@/components/ui/Spinner";
import {
  PIECE_LABELS,
  fetchDatasetSamples,
  relabelSample,
  type DatasetSample,
} from "@/lib/api/dataset";

type Props = {
  jobId: string;
};

export function DatasetReviewPanel({ jobId }: Props) {
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyPath, setBusyPath] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDatasetSamples(jobId);
      setSamples(data.samples);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dataset");
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    void load();
  }, [load]);

  async function handleRelabel(sample: DatasetSample, newLabel: string) {
    if (newLabel === sample.label) return;
    setBusyPath(sample.path);
    try {
      await relabelSample(jobId, sample.path, newLabel);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Relabel failed");
    } finally {
      setBusyPath(null);
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-16">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Dataset review</h1>
        <p className="text-sm text-muted-foreground">
          Job <code className="font-mono">{jobId}</code> — correct labels for retraining.
        </p>
      </div>

      {error && <p className="text-sm text-red-500">{error}</p>}

      {samples.length === 0 ? (
        <Card className="p-6 text-sm text-muted-foreground">
          No samples yet. Run a scan with dataset mode enabled.
        </Card>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {samples.map((sample) => (
            <Card key={sample.path} className="p-3 space-y-2">
              <div className="aspect-square rounded-md bg-muted flex items-center justify-center text-xs text-muted-foreground">
                {sample.path.split("/").pop()}
              </div>
              <p className="text-xs font-mono truncate" title={sample.path}>
                {sample.label}
              </p>
              <select
                className="w-full rounded border bg-background px-2 py-1 text-sm"
                value={sample.label}
                disabled={busyPath === sample.path}
                onChange={(e) => void handleRelabel(sample, e.target.value)}
              >
                {PIECE_LABELS.map((label) => (
                  <option key={label} value={label}>
                    {label}
                  </option>
                ))}
              </select>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
