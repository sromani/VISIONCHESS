# Architecture

VisionChess is a monorepo with three deployable apps sharing chess logic and types.

## High-level diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Client apps                          │
├──────────────────────────┬──────────────────────────────────┤
│   apps/web (Next.js)     │   apps/mobile (Vite + Capacitor) │
│   Browser / desktop      │   iOS / Android                  │
└────────────┬─────────────┴──────────────┬───────────────────┘
             │                            │
             │  POST /detect-lc2fen       │  offline stub (Phase 4: WASM)
             ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│              apps/api — FastAPI vision server               │
│  LC2FEN rectify → YOLO bbox → classify → validate → FEN     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│         ml/ — ONNX models, training, inference utils          │
└─────────────────────────────────────────────────────────────┘

Both clients run Stockfish WASM locally (Web Worker) — no engine server.
```

## Web client (`apps/web`)

| Module | Path | Role |
|--------|------|------|
| Pages | `src/app/` | Next.js App Router |
| Board | `src/components/board/` | react-chessboard + chess.js |
| Analysis | `src/components/analysis/` | Eval bar, engine arrows |
| Engine | `src/lib/chess/stockfishEngine.ts` | Singleton WASM worker |
| State | `src/store/appStore.ts` | Zustand — FEN, history, analysis |
| Storage | `src/lib/storage/` | localStorage board snapshots |
| API client | `src/lib/api/` | Vision pipeline HTTP calls |

**Dev proxy:** Next.js rewrites `/api/v1` → `API_BACKEND_URL` (default `127.0.0.1:8001`).

## Mobile client (`apps/mobile`)

Same chess/analysis modules as web, adapted for Vite:

- `import.meta.env` instead of `process.env`
- Capacitor Camera for photo/gallery
- Offline pipeline stub until on-device WASM vision ships
- Native projects: `android/`, `ios/` (Capacitor-generated, tracked in Git)

## Vision API (`apps/api`)

```
apps/api/vision/
├── lc2fen/           # Board localization & rectification
├── yolo/             # Piece bounding boxes
├── classify/         # Square occupancy + piece labels
├── validate/         # FEN validation, castling inference
└── routes/           # FastAPI endpoints
```

Entry: `POST /api/v1/detect-lc2fen` — multipart image upload → `DetectionResult` JSON.

## ML layer (`ml/`)

| Directory | Purpose |
|-----------|---------|
| `inference/` | ONNX runtime wrappers |
| `training/` | Classifier training CLIs |
| `models/` | Weights (gitignored) + JSON configs |
| `vendor/` | chesscog, LiveChess2FEN reference code |

## Stockfish integration

```
UI change (move / FEN)
  → useStockfishAnalysis hook (debounce 250ms, abort previous)
  → stockfishEngine.analyze(fen, { depth: 12, multiPv: N })
  → Worker: position fen … / go depth 12
  → UCI parse → AnalysisResult → Zustand store → eval bar + arrows
```

Worker script served from `/stockfish.wasm.js` (copied on `npm postinstall`).

## Storage

| Data | Mechanism | Location |
|------|-----------|----------|
| Board snapshots | localStorage | `visionchess:board-snapshots` |
| Engine settings | Zustand (in-memory) | showEngineArrows, engineMultiPv |
| Mobile preferences | Capacitor Preferences (future) | settings migration planned |

## Security & deployment

- No secrets in Git — use `.env` (gitignored)
- Model weights excluded — download via `ml/scripts/setup_pretrained`
- CORS configured via `CORS_ORIGINS` in API `.env`
- Mobile: 100% offline by default (no `VITE_API_URL`)

See also: [vision-pipeline.md](vision-pipeline.md), [mobile.md](mobile.md), [CHESS_VISION_ARCHITECTURE.md](CHESS_VISION_ARCHITECTURE.md).
