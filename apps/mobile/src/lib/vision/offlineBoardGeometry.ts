import { GEOMETRY_BACKEND } from "@/lib/config";
import { buildOfflineDebugImages, rgbaToDataUrl } from "@/lib/vision/drawDebug";
import {
  GEOMETRY_DEBUG_WARP_FAIL_THRESHOLD,
  type GeometryTrace,
} from "@/lib/vision/geometryDebugTypes";
import { archiveGeometryCase } from "@/lib/vision/geometryFailureArchive";
import {
  snapshotSourceDebug,
  snapshotWarpDebug,
} from "@/lib/vision/geometryDebugRender";
import { geoDebugLog } from "@/lib/vision/geometryDebugLogger";
import {
  loadImageDataFromFile,
  rectifyBoardFast,
} from "@/lib/vision/fastBoardGeometry";
import { resetOpenCvWorker, runOpenCvRectify } from "@/lib/vision/opencvLoader";
import type { CornerPoint } from "@/lib/vision/cornerOrder";
import type { DetectionDebug, Point, WarpQualityMetrics } from "@/types";

export interface OfflineGeometryResult {
  corners: Point[];
  homography: number[][];
  warpedUrl: string;
  debug: DetectionDebug;
  metadata: Record<string, unknown>;
  originalWidth: number;
  originalHeight: number;
  outputWidth: number;
  outputHeight: number;
  confidence: number;
  processingMs: number;
  warpQuality: WarpQualityMetrics;
}

async function maybeArchiveFailure(params: {
  file: File;
  trace: GeometryTrace;
  source: ImageData;
  warpedRgba: Uint8ClampedArray;
  warpedWidth: number;
  warpedHeight: number;
  originalJpeg: string;
  warpQuality: WarpQualityMetrics;
}): Promise<string | null> {
  if (params.warpQuality.warp_quality_score >= GEOMETRY_DEBUG_WARP_FAIL_THRESHOLD) {
    return null;
  }
  const toggles = {
    showContours: true,
    showCorners: true,
    showGrid: true,
    showWarp: true,
    showRejected: true,
  };
  const sourceOverlay = snapshotSourceDebug(params.source, params.trace, toggles);
  const { warped, grid } = snapshotWarpDebug(
    params.warpedRgba,
    params.warpedWidth,
    params.warpedHeight,
    true,
  );
  const id = await archiveGeometryCase({
    warpQualityScore: params.warpQuality.warp_quality_score,
    failureReason: "low_warp_quality",
    fileName: params.file.name,
    originalJpeg: params.originalJpeg,
    sourceOverlayJpeg: sourceOverlay,
    warpedJpeg: warped,
    gridJpeg: grid,
    trace: params.trace,
  });
  geoDebugLog(
    "offlineBoardGeometry.ts:archive",
    "auto_archived_failure",
    { id, score: params.warpQuality.warp_quality_score },
    "H-archive",
  );
  return id;
}

function packResult(params: {
  file: File;
  started: number;
  source: ImageData;
  corners: Point[];
  homography: number[][];
  warpedRgba: Uint8ClampedArray;
  warpedWidth: number;
  warpedHeight: number;
  geometryTrace: GeometryTrace;
  warpQuality: WarpQualityMetrics;
  backend: string;
  contourPolygon: CornerPoint[];
}): Promise<OfflineGeometryResult> {
  const {
    file,
    started,
    source,
    corners,
    homography,
    warpedRgba,
    warpedWidth,
    warpedHeight,
    geometryTrace,
    warpQuality,
    backend,
    contourPolygon,
  } = params;

  const debugImages = buildOfflineDebugImages({
    source,
    contourPolygon,
    corners,
    rawCorners: geometryTrace.rawCorners,
    orderedCorners: geometryTrace.orderedCorners,
    candidates: geometryTrace.candidates,
    allContourTraces: geometryTrace.contours,
    warpedRgba,
    warpedWidth,
    warpedHeight,
    warpQualityScore: warpQuality.warp_quality_score,
  });

  const originalJpeg = rgbaToDataUrl(source.data, source.width, source.height);
  const warpedUrl =
    debugImages.rectifiedBoard ??
    rgbaToDataUrl(warpedRgba, warpedWidth, warpedHeight);

  const debug: DetectionDebug = {
    original: originalJpeg,
    boardContour: debugImages.contourOverlay,
    rectifiedBoard: debugImages.rectifiedBoard,
    rectifiedGrid: debugImages.rectifiedGrid,
    cornersOriginal: debugImages.cornersOriginal,
    cornersWarped: debugImages.cornersWarped,
    detectedLines: debugImages.allContours,
    intersections: debugImages.candidatesOverlay,
    mesh: debugImages.rawCornersOverlay,
    rectifiedPreprocessed: debugImages.orderedCornersOverlay,
  };

  return (async () => {
    let archiveId: string | null = null;
    try {
      archiveId = await maybeArchiveFailure({
        file,
        trace: geometryTrace,
        source,
        warpedRgba,
        warpedWidth,
        warpedHeight,
        originalJpeg,
        warpQuality,
      });
    } catch {
      /* ignore */
    }

    const processingMs = Math.round(performance.now() - started);
    const selectedCand = geometryTrace.candidates.find((c) => c.selected);
    geoDebugLog(
      "offlineBoardGeometry.ts:done",
      "rectify_done",
      {
        processingMs,
        backend,
        score: warpQuality.warp_quality_score,
        selectedPass: selectedCand?.pass ?? null,
      },
      "H-pipeline",
    );

    return {
      corners,
      homography,
      warpedUrl,
      debug,
      originalWidth: source.width,
      originalHeight: source.height,
      outputWidth: warpedWidth,
      outputHeight: warpedHeight,
      confidence: Math.min(0.99, warpQuality.warp_quality_score / 100),
      processingMs,
      warpQuality,
      metadata: {
        offline: true,
        geometryOnly: true,
        geometry_backend: backend,
        geometry_trace: geometryTrace,
        warp_quality: warpQuality,
        warp_quality_score: warpQuality.warp_quality_score,
        homography,
        warp_preprocess: "none",
        board_ready: false,
        fen_valid: false,
        failure_archived: !!archiveId,
        failure_archive_id: archiveId,
      },
    };
  })();
}

async function rectifyBoardFastPath(file: File): Promise<OfflineGeometryResult> {
  const started = performance.now();
  geoDebugLog("offlineBoardGeometry.ts:start", "rectify_fast", { file: file.name }, "H-pipeline");

  const source = await loadImageDataFromFile(file);
  const fast = rectifyBoardFast(source);

  return packResult({
    file,
    started,
    source,
    corners: fast.corners,
    homography: fast.homography,
    warpedRgba: fast.warpedRgba,
    warpedWidth: fast.warpedWidth,
    warpedHeight: fast.warpedHeight,
    geometryTrace: fast.trace,
    warpQuality: fast.warpQuality,
    backend: "canvas_fast",
    contourPolygon: fast.trace.selectedContour,
  });
}

async function rectifyBoardOpenCvPath(file: File): Promise<OfflineGeometryResult> {
  const started = performance.now();
  geoDebugLog("offlineBoardGeometry.ts:start", "rectify_opencv", { file: file.name }, "H-pipeline");

  const source = await loadImageDataFromFile(file, 960);

  let workerResult;
  try {
    workerResult = await runOpenCvRectify(source.data, source.width, source.height);
  } catch (err) {
    resetOpenCvWorker();
    throw err;
  }

  const geometryTrace = workerResult.geometryTrace;
  if (!geometryTrace) {
    throw new Error("Geometry trace missing — reload and retry");
  }

  const warpedRgba = new Uint8ClampedArray(workerResult.warpedRgba.slice(0));
  const corners: Point[] = workerResult.corners.map(([x, y]) => ({ x, y }));
  const contourPolygon: CornerPoint[] = workerResult.contourPolygon.map(([x, y]) => ({
    x,
    y,
  }));
  const warpQuality = workerResult.warpQuality as WarpQualityMetrics;

  return packResult({
    file,
    started,
    source,
    corners,
    homography: workerResult.homography,
    warpedRgba,
    warpedWidth: workerResult.warpedWidth,
    warpedHeight: workerResult.warpedHeight,
    geometryTrace,
    warpQuality,
    backend: "opencv_js_worker",
    contourPolygon,
  });
}

/**
 * Offline board geometry — default `fast` (&lt;3s). Optional `VITE_GEOMETRY_BACKEND=opencv`.
 */
export async function rectifyBoardFromFile(file: File): Promise<OfflineGeometryResult> {
  if (GEOMETRY_BACKEND === "opencv") {
    return rectifyBoardOpenCvPath(file);
  }
  return rectifyBoardFastPath(file);
}
