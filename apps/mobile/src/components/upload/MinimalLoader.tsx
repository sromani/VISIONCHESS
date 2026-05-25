import { Spinner } from "@/components/ui/Spinner";

export function MinimalLoader({ label = "Reading position…" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-24">
      <Spinner label={label} />
    </div>
  );
}
