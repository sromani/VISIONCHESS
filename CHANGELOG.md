# Changelog

All notable changes to VisionChess are documented here.

## [0.1.0] — 2026-05-25

### Added

- **Web app** (`apps/web`) — photo upload, LC2FEN + YOLO board detection, interactive legal board
- **Stockfish analysis** — client-side WASM engine, eval bar, MultiPV arrows, move history
- **Mobile app** (`apps/mobile`) — Capacitor iOS/Android, offline-first, camera & gallery
- **Vision API** (`apps/api`) — board localization, piece detection, FEN validation
- **ML pipeline** (`ml/`) — occupancy + piece classifiers, YOLO ONNX inference
- **Benchmark suite** — real-photo accuracy metrics
- **Docker Compose** — optional Postgres, Redis, API + web services

### Notes

- ML model weights must be downloaded separately (`ml/models/README.md`)
- Mobile on-device vision (OpenCV + YOLO WASM) planned for a future release
