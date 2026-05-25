# VisionChess branding

## Source

| File | Purpose |
|------|---------|
| `source-app-icon.png` | Regenerated master (1024×1024, **no text**, tight crop) |
| `archive/source-app-icon-with-text.jpg` | Original artwork with wordmark (backup) |
| `app-icon-master.png` | Same as processed master |
| `previews/` | Simulated iOS / Android / small-size previews |

The generator **strips “VisionChess” text**, crops to knight + lens + reticle, and scales the mark to **~88%** of the canvas (square icons) and **~80%** for Android adaptive foreground (transparent PNG).

## Regenerate

```bash
cd apps/mobile
npm run generate:branding
npm run cap:sync
```

## Outputs

- `public/branding/`, `public/icons/`, favicons
- iOS `Assets.xcassets` (AppIcon + Splash)
- Android `ic_launcher*.png` + `ic_launcher_foreground.png` (transparent) + splash

## Tuning

Edit `scripts/generate-branding-assets.cjs`:

- `ICON_FILL` — square / iOS / PWA (default `0.96`, cover crop)
- `ADAPTIVE_FOREGROUND_FILL` — Android foreground (default `0.88`)
- `MASKABLE_FILL` — PWA maskable safe inset (default `0.82`)

Android adaptive background: `#000000` (`ic_launcher_background`).
