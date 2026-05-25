# Mobile app

Offline-first chess analysis for **iOS** and **Android** using Capacitor.

## Stack

| Layer | Technology |
|-------|------------|
| UI | React 19 + Vite 6 + Tailwind |
| Native shell | Capacitor 7 |
| Chess | chess.js + react-chessboard |
| Engine | Stockfish WASM (Web Worker) |
| Camera | `@capacitor/camera` |
| Storage | localStorage (Capacitor Preferences planned) |

## Project layout

```
apps/mobile/
├── src/                 # React app (shared patterns with web)
├── public/              # Static assets + Stockfish (postinstall)
├── android/             # Android Studio project (tracked)
├── ios/                 # Xcode project (tracked)
├── capacitor.config.ts  # App ID, splash, plugins
└── dist/                # Web build output (gitignored)
```

## App identity

| Setting | Value |
|---------|--------|
| Name | VisionChess |
| Bundle ID | `com.visionchess.app` |
| Web dir | `dist/` |

## Development

```bash
cd apps/mobile
npm install
npm run dev              # http://localhost:5173
```

Browser dev uses file picker fallback for camera/gallery.

## Native builds

### Android

Requires [Android Studio](https://developer.android.com/studio).

```bash
npm run build
npm run cap:sync
npm run cap:open:android
```

Run from Android Studio on emulator or device.

### iOS (macOS only)

Requires Xcode + CocoaPods.

```bash
npm run build
npm run cap:sync
npm run cap:open:ios
```

## Offline mode

Default: **no remote API**. The app:

1. Runs Stockfish locally for all analysis
2. Saves boards to localStorage
3. Uses a vision **stub** (starting position) when scanning photos until on-device WASM ships

Optional dev API:

```env
# apps/mobile/.env
VITE_API_URL=http://10.0.2.2:8001/api/v1
```

(`10.0.2.2` = host machine from Android emulator)

## Git handling

| Tracked | Ignored |
|---------|---------|
| `android/` source, Gradle files | `android/build/`, `.gradle/` |
| `ios/` source, Xcode project | `Pods/`, `DerivedData/`, `build/` |
| `capacitor.config.ts` | `dist/`, copied web assets in native dirs |

Capacitor regenerates `app/src/main/assets/public` on `cap sync` — gitignored.

## Icons & splash

1. Place source art in `resources/` (1024×1024 icon, 2732×2732 splash)
2. Generate:

```bash
npx @capacitor/assets generate \
  --iconBackgroundColor '#09090b' \
  --splashBackgroundColor '#09090b'
npm run cap:sync
```

## Permissions

Configured in `capacitor.config.ts`:

- **Camera** — take board photos
- **Photos** — pick from gallery

Android/iOS permission strings are in native project manifests after `cap add`.

## Roadmap

- [x] Capacitor shell + offline Stockfish
- [x] Camera / gallery via Capacitor
- [x] Mobile UI (safe areas, bottom sheets)
- [ ] On-device vision workers (OpenCV + YOLO WASM)
- [ ] Capacitor Preferences for settings
- [ ] App Store / Play Store release
