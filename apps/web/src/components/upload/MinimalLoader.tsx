import { Spinner } from "@/components/ui/Spinner";

export function MinimalLoader() {
  return (
    <div className="flex flex-col items-center justify-center py-24">
      <Spinner label="Reading position…" />
    </div>
  );
}
