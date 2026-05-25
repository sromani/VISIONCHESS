# VisionChess Mobile

Offline-first chess analysis app for **iOS** and **Android** built with **React + Vite + Capacitor**.

## Architecture

| Layer | Stack |
|-------|--------|
| UI | React 19, Tailwind, Zustand |
| Chess | chess.js, react-chessboard |
| Engine | Stockfish WASM (Web Worker) |
| Mobile | Capacitor 7 (Camera, Storage, Splash) |
| Vision | Phase 4 — OpenCV/YOLO WASM workers (stub offline pipeline today) |

Everything runs **on-device**. No backend required.

## Quick start

```bash
cd apps/mobile
npm install
npm run dev          # http://localhost:5173
npm run build
npm run cap:sync     # build + copy to native projects
```

## Native builds

### Android (Windows / macOS / Linux)

Requires [Android Studio](https://developer.android.com/studio) + SDK.

```bash
npm run cap:sync
npm run cap:open:android
# Build & Run from Android Studio
```

Or: `npm run cap:run:android`

### iOS (macOS only)

Requires Xcode.

```bash
npm run cap:sync
npm run cap:open:ios
```

## App identity

| Setting | Value |
|---------|--------|
| App name | VisionChess |
| Bundle ID | `com.visionchess.app` |
| Web dir | `dist/` |

Edit `capacitor.config.ts` to change bundle ID or display name.

## Icons & splash

1. Add source art to `resources/` (1024×1024 icon, 2732×2732 splash).
2. Run `@capacitor/assets` (optional):

```bash
npx @capacitor/assets generate --iconBackgroundColor '#09090b' --splashBackgroundColor '#09090b'
npm run cap:sync
```

## Permissions

Configured via Capacitor plugins:

- **Camera** — take board photos
- **Photos** — pick from gallery
- **Storage** — local board snapshots (localStorage / IndexedDB)

Android/iOS permission strings are in `android/` and `ios/` after `cap add`.

## Optional dev API

For remote vision during development only:

```env
VITE_API_URL=http://10.0.2.2:8001/api/v1
```

Leave unset for **100% offline** mode (default).

## Roadmap

- [x] Phase 1 — Capacitor shell, Vite app, native projects
- [x] Phase 2 — Mobile UI, safe areas, bottom sheets
- [x] Phase 3 — Camera + gallery (Capacitor Camera)
- [ ] Phase 4 — OpenCV + YOLO in dedicated workers
- [x] Phase 5–6 — Offline storage (localStorage snapshots)
- [x] Phase 7 — Dark mode, adaptive layout, store-ready config

## Monorepo

| App | Purpose |
|-----|---------|
| `apps/web` | Next.js web + optional Python API |
| `apps/mobile` | Capacitor mobile (this app) |
