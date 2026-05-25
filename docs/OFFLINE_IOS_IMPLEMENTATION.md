# Offline iOS implementation plan

**Goal:** On iPhone, without a server, run the same conceptual pipeline as online `/detect-lc2fen`:

1. Board geometry (LC2FEN-style in Phase 2; interim: existing mobile warp)
2. Top-down rectified board
3. `yolov8_chess_pieces.onnx` inference
4. Square assignment + FEN + validation
5. Return result to the Capacitor WebView UI

**Out of scope (explicit):** new models, retraining, threshold tuning, alternative detectors (chess-vision, photochess), canvas geometry as final solution, changes to the online API path.

---

## 1. Current state (audit)

### 1.1 Where offline warp lives today

| Layer | File(s) | Backend |
|-------|---------|---------|
| Entry | `apps/mobile/src/lib/pipeline.ts` → `runLocalGeometryPipeline()` | `rectifyBoardFromFile()` |
| Router | `apps/mobile/src/lib/vision/offlineBoardGeometry.ts` | `GEOMETRY_BACKEND` from `config.ts` |
| **Default fast** | `apps/mobile/src/lib/vision/fastBoardGeometry.ts` | **JS Canvas**: Sobel edges → 4 corners → homography → `warpPerspective` |
| Optional slow | `opencvLoader.ts` + `workers/opencvBoard.worker.ts` + `rectifyBoardCore.ts` | **OpenCV.js WASM** in Web Worker (~minutes on phone) |

Default in dev: `VITE_GEOMETRY_BACKEND` unset → **`fast`** (`canvas_fast`).

### 1.2 How the rectified board is built (fast path)

1. `loadImageDataFromFile(file, maxEdge=640)` — decode + downscale via `createImageBitmap` + Canvas 2D.
2. Grayscale → box blur → Sobel magnitude.
3. `cornersFromEdges()` or **inset fallback** (76% centered quad if edges fail).
4. `orderCorners()` — TL, TR, BR, BL.
5. `warpSize = clamp(512..640)`; `homographyFromQuad` + `warpPerspective` → `warpedRgba` square.
6. `buildOfflineDebugImages()` → data URLs: `rectifiedBoard`, `rectifiedGrid`, corner overlays.

Metadata: `geometry_backend: "canvas_fast"`, `geometryOnly: true`, `board_ready: false`, empty `squares[]`.

### 1.3 What the API does instead (reference)

| Step | Online (`apps/api`) | Offline mobile today |
|------|---------------------|----------------------|
| Geometry | `vision/lc2fen/geometry.py` → `detect_input_board()` (LAPS + OpenCV in `ml/vendor/LiveChess2FEN`) | Canvas Sobel / optional OpenCV WASM |
| Rectified image | `rectified_bgr` from LC2FEN tmp + `compute_perspective_warp` | Canvas RGBA → JPEG data URL |
| Pieces | `yolo_pieces.py` → `get_yolo_piece_classifier()` on **rectified BGR** | **None** (unless `VITE_API_URL` set → hybrid calls server) |
| FEN | `_fen_from_assignments` + `fen_validate.py` (python-chess) | Empty FEN |

**Root cause of offline failure:** warp often wrong on real iPhone/WhatsApp photos; **no on-device YOLO** even when warp is acceptable.

### 1.4 JS / Canvas / WASM split

| Component | Runtime |
|-----------|---------|
| Image decode, warp, debug overlays | **Main thread JS** (Canvas) |
| OpenCV rectify (optional) | **Web Worker** + `@techstark/opencv-js` WASM |
| YOLO / LAPS | **Not on device** |
| FEN UI (`SyntheticBoard`, `detections.ts`) | **JS** (ready once `squares` / `fen` populated) |

---

## 2. Target architecture (Capacitor + native iOS)

```
┌─────────────────────────────────────────────────────────────┐
│  apps/mobile (React + Capacitor WebView)                     │
│  pipeline.ts → runLocalGeometryPipeline()                    │
│    1) rectifyBoardFromFile()  [Phase1: canvas_fast]          │
│    2) VisionChessOffline.recognizeFromWarp()  [native]       │
│    3) map → DetectionResult → UI                             │
└───────────────────────────┬─────────────────────────────────┘
                            │ Capacitor bridge
┌───────────────────────────▼─────────────────────────────────┐
│  packages/visionchess-offline (iOS plugin)                   │
│  Swift + OpenCV (Phase 2) + ONNX Runtime + CoreML EP         │
│                                                              │
│  Phase 1: recognizePieces(warpedJpegBase64)                  │
│  Phase 2: + rectifyFromPhoto(jpegBase64) using LAPS          │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
  yolov8_chess_pieces.onnx      laps_model.onnx + LC2FEN CV
  (bundle Resource)             (Phase 2, vendor Python port)
```

### 2.1 Technology choices (fixed)

| Piece | Choice | Why |
|-------|--------|-----|
| Bridge | **Capacitor 7** custom plugin | Matches `apps/mobile` |
| Inference | **ONNX Runtime iOS** with **CoreML EP** | Same `.onnx` as API; ANE on iPhone |
| Geometry Phase 2 | **OpenCV iOS** + port `detect_input_board` / LAPS | Same as `ml/vendor/LiveChess2FEN` |
| FEN validation Phase 1 | **chess.js** in TS | Already in app; mirrors `fen_validate.py` rules |

Alternative considered: `dust-onnx-capacitor` npm package — valid for Phase 1 spike if we want faster ORT wiring; this doc assumes a **first-party** `visionchess-offline` plugin for full control of debug payloads and Phase 2 geometry in one module.

---

## 3. Assets to reuse (no new models)

| Asset | Path | Use |
|-------|------|-----|
| LC2FEN vendor | `ml/vendor/LiveChess2FEN/` | Phase 2 geometry port reference |
| LAPS ONNX | `ml/vendor/LiveChess2FEN/lc2fen/detectboard/models/laps_model.onnx` | Board detection Phase 2 |
| Piece YOLO ONNX | `ml/models/pretrained/yolov8_chess_pieces.onnx` | Phase 1 + 2 inference |
| YOLO metadata | `ml/models/pretrained/yolov8_chess_pieces.json` | Input size, class names, thresholds |
| Online piece pipeline | `apps/api/vision/inference/yolo_detector.py` | Preprocess/postprocess/NMS spec |
| Square → FEN | `apps/api/vision/lc2fen/yolo_pieces.py` | `YOLO_TO_FEN`, `_fen_from_assignments` |
| Square assignment | `apps/api/vision/scanner/square_assignment.py` | `center_to_square_name`, one piece per square |
| FEN validate | `apps/api/vision/lc2fen/fen_validate.py` | `board_ready` / `interactive_fen` rules |
| Geometry API | `apps/api/vision/lc2fen/geometry.py` | `rectify_board_from_bytes` contract |
| Mobile FEN labels | `apps/mobile/src/lib/chess/detections.ts` | `LABEL_TO_FEN`, `detectionsFromSquares` |

**Do not use for offline pieces:** `MobileNetV2_0p5_all.onnx` (64-tile LC2FEN classifier) — production endpoint uses **YOLO**, not MobileNet.

---

## 4. What runs where

### Phase 1

| Step | Native (Swift) | TypeScript |
|------|----------------|------------|
| Pick photo / `File` | — | `camera.ts`, `appStore.upload` |
| Decode + warp | — | `fastBoardGeometry.ts` (unchanged) |
| Debug geometry images | — | `drawDebug.ts`, `geometryDebugLogger.ts` |
| RGB pad 640, YOLO ORT | **Yes** | — |
| NMS, square assignment, placement FEN | **Yes** (or TS mirror for tests) | Map to `DetectionResult` |
| `interactive_fen`, `board_ready` | — | **chess.js** (`fenFromYolo.ts`) |
| Synthetic board UI | — | `DetectedPositionBoard`, `SyntheticBoard` |
| Persist debug bundle | Optional write to app cache | `offlineDebugStore.ts` |

### Phase 2

| Step | Native (Swift) | TypeScript |
|------|----------------|------------|
| Full pipeline from photo bytes | **LAPS + OpenCV warp** | Call `VisionChessOffline.rectifyAndRecognize()` |
| YOLO + FEN | **Yes** | Same mapping as Phase 1 |
| Deprecate | — | `fastBoardGeometry` as production path; keep for A/B debug only |

---

## 5. Capacitor plugin API (concrete)

Package: `packages/visionchess-offline/`

### 5.1 Methods

```typescript
// Phase 1
recognizeFromWarpedJpeg(options: {
  jpegBase64: string;
  width: number;
  height: number;
  confThreshold?: number; // default 0.30 — match API query
}): Promise<OfflineRecognizeResult>;

// Phase 2
rectifyAndRecognize(options: {
  jpegBase64: string;
  a1Pos?: 'BL' | 'BR' | 'TL' | 'TR';
  confThreshold?: number;
}): Promise<OfflineFullPipelineResult>;
```

### 5.2 `OfflineRecognizeResult` (returned to JS)

```typescript
{
  placementFen: string;           // e.g. "rnbqkbnr/pppppppp/8/8/..."
  squares: Array<{
    name: string;                 // "e4"
    label: string;                // "white_pawn" | "empty"
    confidence: number;
    occupied: boolean;
    bbox: [number, number, number, number];
  }>;
  detections: Array<{ label; confidence; bbox; square }>;
  timings: {
    preprocessMs: number;
    inferenceMs: number;
    postprocessMs: number;
    totalMs: number;
  };
  debug: {
    warpedJpegBase64?: string;    // echo input
    overlayJpegBase64: string;   // boxes + square labels (port yolo_viz)
    logLines: string[];
  };
}
```

TS then sets `fenValid`, `boardReady`, `interactiveFen` via chess.js (same thresholds as `fen_validate.py`: min 4 pieces, kings ok).

### 5.3 Loading ONNX on iPhone

1. Copy `ml/models/pretrained/yolov8_chess_pieces.onnx` → `packages/visionchess-offline/ios/Resources/yolov8_chess_pieces.onnx` (script: `apps/mobile/scripts/copy-offline-models.cjs`).
2. Xcode: add to **Copy Bundle Resources**.
3. Swift: `ORTEnv` + `ORTSession` with session options:
   - `appendCoreMLExecutionProvider` (CoreML EP, `CPUAndNeuralEngine` on device).
   - Fallback CPU EP.
4. Input tensor: `[1, 3, 640, 640]` float32, RGB, `/255`, letterbox pad 114 — copy `_preprocess()` from `yolo_detector.py` exactly.
5. Output: `output0` — decode per `_postprocess()` / `_reshape_predictions()`.

**Model size:** ~50–80 MB; acceptable for dev builds; use On-Demand Resources later if needed.

---

## 6. Phase 1 — minimal functional spike

### 6.1 Scope

- Keep **`rectifyBoardFromFile` → `canvas_fast`**.
- After warp, pass **rectified JPEG base64** to native plugin.
- Run **only** `yolov8_chess_pieces.onnx`.
- Return squares + placement FEN + overlay + timings.
- **No server** (`VITE_API_URL` empty).

### 6.2 Files to add / touch

| Action | Path |
|--------|------|
| **New** | `docs/OFFLINE_IOS_IMPLEMENTATION.md` (this file) |
| **New** | `packages/visionchess-offline/` (Capacitor plugin) |
| **New** | `apps/mobile/src/lib/vision/offlineYolo/fenFromYolo.ts` |
| **New** | `apps/mobile/src/lib/vision/offlineYolo/mapNativeResult.ts` |
| **New** | `apps/mobile/src/lib/vision/offlineYolo/nativeBridge.ts` |
| **New** | `apps/mobile/scripts/copy-offline-models.cjs` |
| **Edit** | `apps/mobile/src/lib/pipeline.ts` — call native after geo when `Capacitor.isNativePlatform()` |
| **Edit** | `apps/mobile/package.json` — `"@visionchess/offline": "file:../../packages/visionchess-offline"` |
| **Edit** | `apps/mobile/capacitor.config.ts` — register plugin |
| **No change** | `apps/api/**`, online detect-lc2fen |

### 6.3 Pipeline change (Phase 1)

```typescript
// pipeline.ts — after rectifyBoardFromFile()
if (Capacitor.isNativePlatform()) {
  const native = await recognizeFromWarpedNative(geo);
  detection = mergeGeometryWithNativePieces(geo, native, originalUrl);
} else {
  // browser dev: geometryOnly + banner "YOLO requires iOS build"
}
```

### 6.4 Debug / logging (required)

| Artifact | Where |
|----------|--------|
| Structured log lines | `metadata.offline_log[]` + `geometryDebugLogger` |
| Warped board dump | `debug.rectifiedBoard` (existing) |
| YOLO overlay | `debug.mlPieceTop1` or `debug.yoloOverlay` |
| Timings | `metadata.timings: { geometryMs, yoloPreMs, yoloInferMs, yoloPostMs }` |
| Intermediate screenshots | Capacitor `Filesystem` → `offline-debug/<jobId>/` |

Enable with `VITE_OFFLINE_DEBUG=1` (no impact on online).

---

## 7. Phase 2 — LC2FEN native geometry

### 7.1 Scope

Replace Canvas warp with native port of:

1. `lc2fen.predict_board.detect_input_board` (uses `laps_model.onnx` + OpenCV in vendor).
2. `vision.lc2fen.common.compute_perspective_warp`.
3. Same YOLO path as Phase 1 on `rectified_bgr`.

### 7.2 Swift port sources (read-only reference → C++/Swift)

| Python module | Vendor path |
|---------------|-------------|
| Board detect | `ml/vendor/LiveChess2FEN/lc2fen/detectboard/` |
| LAPS inference | `lc2fen/detectboard/models/laps_model.onnx` |
| Warp | `ml/vendor/LiveChess2FEN/lc2fen/predict_board.py`, `vision/lc2fen/common.py` |

**Approach:** Run Phase 2 as a **thin native wrapper** calling vendored Python is **not** acceptable on iOS App Store; must port logic to Swift + OpenCV + ORT (LAPS ONNX).

### 7.3 Acceptance

Side-by-side on 20 fixture photos: native Phase 2 warp vs API `rectified_board` SSIM/corners delta; then same YOLO → FEN as online.

---

## 8. Viability, blockers, milestones

### 8.1 Is it technically viable?

**Yes.** The repo already contains the ONNX model, the exact Python postprocessing, and a Capacitor iOS shell. The work is **integration engineering**, not research.

### 8.2 What blocks today

| Blocker | Severity | Mitigation |
|---------|----------|------------|
| No native plugin | **Critical** | Phase 1 plugin + ORT SPM |
| YOLO only in Python API | **Critical** | Port `_preprocess` / `_postprocess` to Swift (spec in TS for tests) |
| Wrong warp (`canvas_fast`) | **High** for FEN accuracy | Phase 1 still proves YOLO path; Phase 2 fixes geometry |
| OpenCV WASM too slow | Medium | Do not use WASM for production offline |
| ONNX in WebView / Expo | Medium | Native only; no `onnxruntime-web` on phone |
| Model bundle size | Low | Embed in app; later ODR |
| `chess.js` vs `python-chess` validation edge cases | Low | Accept small parity diffs in Phase 1 |

### 8.3 Milestones (realistic)

| # | Milestone | Done when | ETA (1 dev) |
|---|-----------|-----------|-------------|
| **M0** | Doc + TS FEN mapper + plugin skeleton | Plugin registers; TS unit tests for FEN from mock detections | 2–3 days |
| **M1** | **Phase 1 on device:** warp (existing) + native YOLO → placement FEN + overlay | iPhone photo → synthetic board with pieces (even if warp imperfect) | 1–2 weeks |
| **M2** | Debug bundle (timings, dumps, logs) on device | Folder or panel shows all intermediates | +3–5 days |
| **M3** | Phase 2 LAPS warp on device | Warp visually matches API on majority of fixture set | 3–5 weeks |
| **M4** | Offline parity with online on benchmark set | Same FEN on ≥70% of `real_photo` fixtures (no threshold tuning) | after M3 |

**First realistic user-visible win:** **M1** — FEN from on-device YOLO on current warp.

---

## 9. Decision summary

| Question | Answer |
|----------|--------|
| Abandon LC2FEN? | **No.** Phase 2 **is** LC2FEN geometry on device. |
| Abandon current offline warp forever? | **Yes for production** after Phase 2; keep for Phase 1 spike only. |
| Integrate first | **YOLO ONNX native (Phase 1)** on existing warp |
| Copy from | `yolo_detector.py`, `yolo_pieces.py`, `square_assignment.py`, `yolov8_chess_pieces.json` |
| Do not copy now | chess-vision, photochess, chessml, ARChessAnalyzer pipelines |

**Best realistic option today:** **LC2FEN + existing YOLO ONNX**, implemented as a **Capacitor Swift plugin** with ORT CoreML EP — Phase 1 pieces-only, Phase 2 native LAPS geometry.

---

## 10. iOS build checklist

```bash
# From repo root
node apps/mobile/scripts/copy-offline-models.cjs
cd apps/mobile
npm install
npm run build
npx cap sync ios
npx cap open ios
# Xcode: add onnxruntime-swift-package-manager, OpenCV2 (Phase 2), verify Resources model
# Run on physical iPhone (Simulator CoreML behavior differs)
```

Env for offline-only dev:

```env
# apps/mobile/.env.development
VITE_API_URL=
VITE_OFFLINE_NATIVE_YOLO=true
VITE_OFFLINE_DEBUG=1
```

---

## 11. Related docs

- Online reference (frozen): `docs/MOBILE_LC2FEN_PIPELINE.md`
- Prior research comparison: (conversation) — do not fork alternate OSS pipelines per this plan.
