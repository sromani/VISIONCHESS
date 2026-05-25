# Board detection benchmark

Fixed image set for regression checks before changing localization heuristics.

## Setup

1. Copy 10–20 JPEG/PNG photos into `images/` (real scans that used to pass or fail).
2. Edit `manifest.json`:

```json
{
  "baselineMode": "stable_v1",
  "images": ["board_center.jpg", "angled_table.jpg"]
}
```

3. Enable dev UI: `VITE_DEV_MODE=true` in `.env`, rebuild, open the app → **Board detection benchmark**.

## Modes (`.env`)

| `VITE_DETECTION_MODE` | Behavior |
|----------------------|----------|
| `stable_v1` (default) | YOLO board bbox → direct warp; else single contour+LAPS / Hough |
| `yolo_v1` | YOLO board only — no OpenCV overwrite |
| `experimental` | Multi-candidate search (debug / A-B only) |
| `mesh_v2` | Not implemented — uses `stable_v1` |

## Failures

Failed scans are stored in `localStorage`. Use **Export failures** in the benchmark panel, then unpack into `debug_failures/` if you want files on disk:

```bash
mkdir -p debug_failures
# save downloaded debug_failures_*.json as debug_failures/export.json
node scripts/unpack-debug-failures.cjs debug_failures/export.json
```

Each case includes: original image, overlay, candidates, scores, rejection reason.
