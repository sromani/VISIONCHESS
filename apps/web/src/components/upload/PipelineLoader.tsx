"use client";

import { cn } from "@/lib/utils";
import { PipelineStepState } from "@/types";

export function PipelineLoader({
  steps,
  fileName,
}: {
  steps: PipelineStepState[];
  fileName: string | null;
}) {
  return (
    <div className="animate-fade-up mx-auto w-full max-w-lg">
      <div className="glass rounded-2xl p-8">
        <p className="text-center text-sm text-muted">Processing</p>
        <h2 className="mt-1 text-center text-lg font-medium tracking-tight">
          {fileName ?? "your image"}
        </h2>

        <ol className="mt-8 space-y-4">
          {steps.map((step) => (
            <li key={step.id} className="flex items-center gap-3">
              <StepIcon status={step.status} />
              <span
                className={cn(
                  "text-sm transition-colors",
                  step.status === "active" && "font-medium text-foreground",
                  step.status === "done" && "text-muted",
                  step.status === "pending" && "text-muted/60",
                )}
              >
                {step.label}
              </span>
              {step.status === "active" && (
                <span className="ml-auto h-1.5 w-16 overflow-hidden rounded-full bg-card-hover">
                  <span className="block h-full w-1/2 animate-[shimmer_1s_ease_infinite] rounded-full bg-accent" />
                </span>
              )}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

function StepIcon({ status }: { status: PipelineStepState["status"] }) {
  if (status === "done") {
    return (
      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent/20 text-xs text-accent">
        ✓
      </span>
    );
  }
  if (status === "active") {
    return (
      <span className="flex h-6 w-6 items-center justify-center">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      </span>
    );
  }
  return <span className="h-6 w-6 rounded-full border border-border" />;
}
