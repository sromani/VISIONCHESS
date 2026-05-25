"use client";

import { DetectedPositionBoard } from "@/components/detection/DetectedPositionBoard";
import { RectifiedResultBoard } from "@/components/detection/RectifiedResultBoard";
import type { DetectionResult } from "@/types";

/** Main scan output: synthetic pieces board + rectified photo warp. */
export function ScanResultBoards({ detection }: { detection: DetectionResult }) {
  const showWarp = Boolean(detection.warpedUrl);

  return (
    <div className="flex w-full max-w-[min(100%,560px)] flex-col gap-5">
      <DetectedPositionBoard detection={detection} />
      {showWarp && <RectifiedResultBoard detection={detection} />}
    </div>
  );
}
