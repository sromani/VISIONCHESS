import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";

interface CaptureActionsProps {
  onCamera: () => void;
  onGallery: () => void;
  onStartPosition: () => void;
  offline?: boolean;
  visionLocal?: boolean;
  className?: string;
}

export function CaptureActions({
  onCamera,
  onGallery,
  onStartPosition,
  offline = true,
  visionLocal = false,
  className,
}: CaptureActionsProps) {
  return (
    <div className={cn("flex w-full flex-col gap-3", className)}>
      <Button size="lg" className="touch-target w-full" onClick={onCamera}>
        Take photo
      </Button>
      <Button size="lg" variant="secondary" className="touch-target w-full" onClick={onGallery}>
        Choose from gallery
      </Button>
      <Button size="lg" variant="ghost" className="touch-target w-full" onClick={onStartPosition}>
        Start from initial position
      </Button>
      {visionLocal && (
        <p className="text-center text-xs leading-relaxed text-muted">
          Offline LC2FEN-style geometry + YOLO → FEN. First scan loads models (~100 MB); allow a few seconds.
        </p>
      )}
      {offline && !visionLocal && (
        <p className="text-center text-xs leading-relaxed text-muted">
          Runs fully offline. Board scan from photo uses on-device vision (Phase 4). Analysis with
          Stockfish is local.
        </p>
      )}
    </div>
  );
}
