import { useEffect, useState } from "react";
import { Capacitor } from "@capacitor/core";
import { SplashScreen } from "@capacitor/splash-screen";

import {
  BOOTSTRAP_FADE_MS,
  BOOTSTRAP_HARD_CAP_MS,
  runAppBootstrap,
  type BootstrapProgress,
} from "@/lib/bootstrap/appBootstrap";
import { useAppStore } from "@/store/appStore";

export type AppBootstrapPhase = "loading" | "fading" | "ready";

export function useAppBootstrap() {
  const [phase, setPhase] = useState<AppBootstrapPhase>("loading");
  const [steps, setSteps] = useState<BootstrapProgress[]>([]);
  const [bootstrapWarning, setBootstrapWarning] = useState<string | null>(null);
  const hydrateSavedBoards = useAppStore((s) => s.hydrateSavedBoards);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      requestAnimationFrame(() => {
        if (Capacitor.isNativePlatform()) {
          void SplashScreen.hide().catch(() => undefined);
        }
      });

      const result = await Promise.race([
        runAppBootstrap((s) => {
          if (!cancelled) setSteps(s);
        }),
        new Promise<Awaited<ReturnType<typeof runAppBootstrap>>>((resolve) => {
          window.setTimeout(() => {
            resolve({
              steps: [],
              warnings: ["Startup took too long — continuing. YOLO loads on first scan."],
            });
          }, BOOTSTRAP_HARD_CAP_MS);
        }),
      ]);

      if (!cancelled && result.warnings.length > 0) {
        setBootstrapWarning(result.warnings[0]);
      }

      if (cancelled) return;

      hydrateSavedBoards();

      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => resolve());
        });
      });

      if (cancelled) return;

      setPhase("fading");
      window.setTimeout(() => {
        if (!cancelled) setPhase("ready");
      }, BOOTSTRAP_FADE_MS);
    })();

    return () => {
      cancelled = true;
    };
  }, [hydrateSavedBoards]);

  return { phase, steps, isReady: phase === "ready", bootstrapWarning };
}
