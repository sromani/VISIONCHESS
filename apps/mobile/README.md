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

Requires Xcode and **CocoaPods** (`pod` on PATH).

```bash
npm run cap:sync
npm run cap:open:ios
```

If you see `CocoaPods is not installed` from the IDE but `pod --version` works in Terminal, add this once to `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$HOME/.gem/ruby/$(ruby -e 'print RUBY_VERSION[/\d+\.\d+/]').0/bin:$PATH"
```

Then `source ~/.zshrc`. `npm run cap:sync` also runs `scripts/with-cocoapods.sh`, which wires the same paths automatically.

Install CocoaPods if needed:

```bash
sudo gem install cocoapods
# or with Homebrew: brew install cocoapods
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

## Web CV lab vs mobile production

| Build | Env | UI |
|-------|-----|-----|
| **Browser lab** | `VITE_WEB_DEBUG=true` | `debug/web/` — side panel, overlays, benchmark |
| **iOS/Android** | no `VITE_WEB_DEBUG` | `mobile/MobileApp` — clean product UI |

```bash
# Web laboratory
cp .env.web.example .env
npm run dev

# Native (production UI)
cp .env.mobile.example .env
npm run build && npm run cap:sync
```

`WEB_DEBUG` is **forced off** on Capacitor native (`IS_NATIVE`), even if the flag is set in `.env`.

Folders: `src/shared/`, `src/vision/`, `src/mobile/`, `src/debug/web/`.

## Board detection modes

```env
VITE_DETECTION_MODE=stable_v1   # default
```

In web lab, change mode in the left panel (re-upload image to apply).

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
