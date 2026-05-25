# VisionChess

Scan a chess board from a photo, get a legal FEN, play moves, and analyze with **Stockfish** — locally on web and mobile.

> Chessvision-style board scanning + Lichess-style analysis board. Offline-first on mobile.

![VisionChess board with Stockfish analysis](docs/screenshots/board-analysis.png)

*Add screenshots to `docs/screenshots/` before publishing.*

---

## Features

| Feature | Web | Mobile |
|---------|-----|--------|
| Photo to FEN pipeline | Yes (API) | On-device WASM planned |
| Interactive legal board | Yes | Yes |
| Stockfish eval + best move | Yes | Yes |
| MultiPV engine arrows | Yes | Yes |
| Move history | Yes | Yes |
| Saved boards | Yes | Yes |
| Camera / gallery | Upload | Capacitor |
| Offline analysis | Yes | Yes |

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Web UI | Next.js 15, React 19, Tailwind, Zustand |
| Mobile | Vite, React 19, Capacitor 7 |
| Chess rules | chess.js |
| Board UI | react-chessboard |
| Engine | stockfish.js WASM Web Worker |
| Vision API | FastAPI, OpenCV, ONNX Runtime |
| Detection | LC2FEN + YOLO piece bbox |

---

## Repository structure

```
VISIONCHESS/
├── apps/web/       Next.js web app
├── apps/mobile/    Capacitor iOS/Android
├── apps/api/       Python FastAPI vision server
├── ml/             Training, inference, models
├── benchmark/      Real-photo accuracy suite
├── docs/           Architecture guides
└── infra/docker/   Dockerfiles
```

| Concept | Location |
|---------|----------|
| UI app | `apps/web/src/app`, `apps/mobile/src/` |
| Components | `apps/*/src/components/` |
| Vision | `apps/api/vision/`, `ml/inference/` |
| Engine | `apps/*/src/lib/chess/stockfishEngine.ts` |
| Mobile native | `apps/mobile/ios/`, `apps/mobile/android/` |
| Models | `ml/models/` (weights gitignored) |

---

## Quick start

### Prerequisites

- Node.js 20+
- Python 3.12+ (vision API)
- Git

### 1. Clone and configure

```bash
git clone https://github.com/sromani/VISIONCHESS.git
cd VISIONCHESS
cp .env.example .env
cp apps/web/.env.local.example apps/web/.env.local
```

### 2. ML models

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m scripts.setup_pretrained
cd ..
```

See [ml/models/README.md](ml/models/README.md).

### 3. Web app

```bash
cd apps/web
npm install
npm run dev
```

Open http://localhost:3000

### 4. Vision API (separate terminal)

```bash
cd apps/api
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### 5. Mobile app

```bash
cd apps/mobile
npm install
npm run dev
npm run build && npm run cap:sync
npm run cap:open:android
```

See [docs/mobile.md](docs/mobile.md).

---

## Stockfish

Both web and mobile run Stockfish in a **Web Worker** (no server):

- `stockfish.js` copied to `public/` on npm postinstall
- Depth 12, MultiPV 1-3, eval bar, engine arrows

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design |
| [docs/vision-pipeline.md](docs/vision-pipeline.md) | Photo to FEN |
| [docs/mobile.md](docs/mobile.md) | Capacitor builds |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

---

## Roadmap

- [x] LC2FEN + YOLO detection
- [x] Interactive board + Stockfish
- [x] Capacitor mobile (offline analysis)
- [ ] On-device vision WASM
- [ ] App Store release

---

## GitHub topics

`chess` `stockfish` `computer-vision` `fen` `react` `nextjs` `capacitor` `onnx` `fastapi`

---

## License

MIT — see [LICENSE](LICENSE). Stockfish WASM is GPL-3.0.
