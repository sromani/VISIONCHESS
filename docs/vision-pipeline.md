# Vision pipeline

End-to-end flow from **chess board photo** to **interactive FEN**.

## Overview

```
Photo (JPEG/PNG)
    │
    ▼
┌─────────────────────┐
│ A. Board localization│  LC2FEN — corner detection, mesh rectify
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ B. Piece detection   │  YOLO ONNX — bounding boxes per piece
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ C. Square assignment │  Map bbox centers → chess squares
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ D. Classification    │  Occupancy + piece CNN (optional refine)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ E. FEN assembly      │  64-square matrix → FEN placement
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ F. Validation        │  python-chess legality, castling inference
└──────────┬──────────┘
           ▼
    interactiveFen + boardReady
```

## API endpoint

```
POST /api/v1/detect-lc2fen
Content-Type: multipart/form-data
Body: file=<image>
```

Response (`DetectionResult`):

| Field | Description |
|-------|-------------|
| `fen` | Raw pipeline FEN (may fail validation) |
| `interactiveFen` | Legal FEN for chess.js board |
| `boardReady` | `true` when interactive play is possible |
| `rectifiedBoardUrl` | Warped board image (base64 data URL) |
| `squares[]` | Per-square labels and confidence |

## Web integration

```typescript
// apps/web/src/lib/pipeline.ts
runVisionPipeline(file, onProgress)
  → detectBoard(file, url, "/detect-lc2fen")
  → appStore.upload() sets history + board state
```

Progress steps shown in UI: upload → detect → classify → validate → analyze.

## Mobile (offline stub)

Until on-device WASM vision ships, `apps/mobile` uses an offline stub:

1. Saves photo preview locally
2. Loads standard starting position
3. Stockfish analysis works immediately on any manually edited position

Future: replace stub with Web Workers running OpenCV + YOLO ONNX in the browser.

## Models

| Stage | Model file | Format |
|-------|------------|--------|
| Piece detection | `yolov8_chess_pieces.onnx` | ONNX |
| Occupancy | `occupancy.onnx` | ONNX |
| Piece classify | `piece_classifier.onnx` | ONNX |

Download: see [ml/models/README.md](../ml/models/README.md).

## Validation highlights

- **Castling rights** inferred from king/rook starting squares when pieces haven't moved (`infer_castling_rights`)
- **Side to move** editable in UI before first move
- Invalid FEN → `boardReady: false`, user prompted to retry photo

## Benchmark

Measure pipeline accuracy on real OTB photos:

```bash
python benchmark/scripts/download_real_photos.py
python benchmark/run_benchmark.py --output benchmark/results/latest.json
```

Metrics: piece accuracy, occupancy F1, legal FEN rate. See [benchmark/README.md](../benchmark/README.md).

## Related docs

- [CHESS_VISION_ARCHITECTURE.md](CHESS_VISION_ARCHITECTURE.md) — ML integration plan (chesscog, LC2FEN)
- [architecture.md](architecture.md) — full system design
