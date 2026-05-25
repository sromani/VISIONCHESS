"use client";

import { useState } from "react";

import { Button } from "@/components/ui/Button";
import { useAppStore } from "@/store/appStore";

export function FenDisplay() {
  const fen = useAppStore((s) => s.fen);
  const [copied, setCopied] = useState(false);

  if (!fen) return null;

  const copy = async () => {
    await navigator.clipboard.writeText(fen);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full max-w-[min(100%,560px)] space-y-2">
      <code className="block break-all rounded-xl border border-border bg-card/50 px-4 py-3 text-center font-mono text-xs leading-relaxed text-foreground">
        {fen}
      </code>
      <div className="flex justify-center">
        <Button variant="ghost" size="sm" onClick={copy}>
          {copied ? "Copied" : "Copy FEN"}
        </Button>
      </div>
    </div>
  );
}
