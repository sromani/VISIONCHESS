# Real-photo benchmark

Labeled OTB photos with expected FEN placement. Used to measure piece accuracy,
occupancy F1, full-board accuracy, and legal FEN rate on **real images** — not
synthetic checkerboards.

## Quick start

```powershell
# 1. Download real OTB photos (chesscog transfer-learning set, 27 images)
python benchmark/scripts/download_real_photos.py

# 2. Build manifest.jsonl from image + .json sidecars
python benchmark/scripts/import_chesscog_transfer_learning.py

# 3. Run benchmark (requires ONNX models — see ml/scripts/export_chesscog_standalone.py)
python benchmark/run_benchmark.py

# Smoke test (1 image)
python benchmark/run_benchmark.py --limit 1

# Full suite + JSON report
python benchmark/run_benchmark.py --output benchmark/results/latest.json
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Piece accuracy** | Correct piece label on occupied squares only |
| **Per-square accuracy** | All 64 squares (piece + empty) |
| **Occupancy F1** | Binary empty vs occupied |
| **Full-board accuracy** | Exact placement match |
| **Legal FEN rate** | `python-chess` valid FEN |

Output includes a **13×13 confusion matrix** (empty + 12 pieces).

## Add your own photos

1. Put image in `benchmark/images/custom/`
2. Add a line to `manifest.jsonl`:

```json
{"id": "my_01", "image": "images/custom/board.jpg", "placement": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQKBR1", "source": "user", "tags": ["real_photo"]}
```

`placement` is the FEN piece field only (8 ranks, `/` separated).

## Pytest

```powershell
cd apps/api
python -m pytest tests/benchmark/test_benchmark_metrics.py -q
python -m pytest tests/benchmark/test_real_photo_benchmark.py -m benchmark -q

# Full 27-image suite (~15 min)
$env:BENCHMARK_FULL=1
python -m pytest tests/benchmark/test_real_photo_benchmark.py -m benchmark -q
```

## Dataset sources

| Folder | Description |
|--------|-------------|
| `images/chesscog_transfer_learning/test/` | 27 real OTB photos, non-standard Staunton set ([chesscog TL dataset](https://github.com/georg-wolflein/chesscog)) |
| `images/custom/` | Your scans (add manually) |

## Results

JSON reports written to `benchmark/results/latest.json` (gitignored).
