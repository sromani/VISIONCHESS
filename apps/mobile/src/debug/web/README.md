# debug/web/

**Browser-only** computer vision laboratory. Tree-shaken out of native builds when `VITE_WEB_DEBUG` is unset.

## Enable

```bash
cp .env.web.example .env
npm run dev
```

## Includes

- Side panel: candidates, YOLO boxes, timings, threshold sliders, benchmark
- Debug overlays on scan (not used on mobile)

## Never import from

- `mobile/MobileApp.tsx`
- Capacitor entry / native shell
