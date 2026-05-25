import { cn } from "@/lib/utils";

export function Spinner({ className, label = "Loading" }: { className?: string; label?: string }) {
  return (
    <div className={cn("flex flex-col items-center gap-3", className)} role="status">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      <span className="text-xs text-muted">{label}</span>
    </div>
  );
}
