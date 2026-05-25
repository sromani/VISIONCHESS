# @visionchess/offline — Capacitor iOS plugin (M1)

Native YOLO ONNX inference on rectified board images.

## Build (Mac + Xcode)

```bash
# From repo root
node apps/mobile/scripts/copy-offline-models.cjs
cd apps/mobile
npm install
npm run build
npx cap sync ios
npx cap open ios
```

Run on a **physical iPhone** (CoreML). First launch loads ~99MB `yolov8_chess_pieces.onnx`.

## Env (`apps/mobile/.env.development`)

```env
VITE_API_URL=
VITE_OFFLINE_DEBUG=1
```

## Debug dumps

With `VITE_OFFLINE_DEBUG=1`, scans save to app data:

`offline-debug/<jobId>/01_warped.jpg`, `02_yolo_overlay.jpg`, `manifest.json`

## Swift sources

| File | Role |
|------|------|
| `YoloChessEngine.swift` | ORT session + CoreML EP |
| `YoloChessPostprocess.swift` | Port of `yolo_detector._postprocess` |
| `YoloChessFen.swift` | Square assignment + placement FEN |
| `YoloChessOverlay.swift` | Debug overlay JPEG |
