import { WEB_DEBUG } from "@/shared/config/flags";
import { useAppStore } from "@/store/appStore";

import { WebDebugSidebar } from "./WebDebugSidebar";
import { WebScanWorkspace } from "./WebScanWorkspace";

/** Browser-only CV laboratory — never mounted on Capacitor. */
export default function WebDebugApp({ bootstrapWarning }: { bootstrapWarning?: string | null }) {
  if (!WEB_DEBUG) {
    return (
      <p className="p-8 text-sm text-muted">
        Set <code className="text-foreground">VITE_WEB_DEBUG=true</code> in .env and run{" "}
        <code className="text-foreground">npm run dev</code> (browser only).
      </p>
    );
  }

  const detection = useAppStore((s) => s.detection);
  const phase = useAppStore((s) => s.phase);
  const pipelineSteps = useAppStore((s) => s.pipelineSteps);

  return (
    <div className="flex h-screen max-h-[100dvh] flex-col bg-background">
      {bootstrapWarning && (
        <div className="shrink-0 border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs text-amber-200/90">
          {bootstrapWarning}
        </div>
      )}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <WebDebugSidebar detection={detection} phase={phase} pipelineSteps={pipelineSteps} />
        <WebScanWorkspace />
      </div>
    </div>
  );
}
