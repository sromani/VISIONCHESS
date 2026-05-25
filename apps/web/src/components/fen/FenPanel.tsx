"use client";

import { useState } from "react";

import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

export function FenPanel() {
  const fen = useAppStore((s) => s.fen);
  const setFen = useAppStore((s) => s.setFen);
  const detection = useAppStore((s) => s.detection);
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(fen);

  const copy = async () => {
    await navigator.clipboard.writeText(fen);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="glass animate-fade-up rounded-2xl p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">FEN</h3>
          <p className="text-xs text-muted">
            {detection?.boardReady
              ? `Validated · ${detection.orientation ?? "normal"}`
              : detection
                ? "Draft FEN — not validated for play"
                : "No position"}
          </p>
        </div>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setDraft(fen);
              setEditing(!editing);
            }}
          >
            {editing ? "Cancel" : "Edit"}
          </Button>
          <Button variant="secondary" size="sm" onClick={copy}>
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
      </div>

      {editing ? (
        <div className="mt-3 space-y-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={3}
            className="w-full resize-none rounded-xl border border-border bg-background p-3 font-mono text-xs leading-relaxed outline-none focus:ring-2 focus:ring-accent/30"
          />
          <Button size="sm" onClick={() => { setFen(draft); setEditing(false); }}>
            Apply
          </Button>
        </div>
      ) : (
        <code
          className={cn(
            "mt-3 block break-all rounded-xl bg-background/60 p-3 font-mono text-xs leading-relaxed",
          )}
        >
          {fen}
        </code>
      )}
    </div>
  );
}
