import { WEB_DEBUG } from "@/shared/config/flags";
import { AppLoadingScreen } from "@/components/splash/AppLoadingScreen";
import { useAppBootstrap } from "@/hooks/useAppBootstrap";
import WebDebugApp from "@/debug/web/WebDebugApp";
import MobileApp from "@/mobile/MobileApp";

export default function AppRoot() {
  const { phase, steps, isReady, bootstrapWarning } = useAppBootstrap();

  const App = WEB_DEBUG ? WebDebugApp : MobileApp;

  return (
    <>
      {isReady && <App bootstrapWarning={bootstrapWarning} />}
      {phase !== "ready" && (
        <AppLoadingScreen steps={steps} exiting={phase === "fading"} warning={bootstrapWarning} />
      )}
    </>
  );
}
