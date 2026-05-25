# ML Models

Model weights are **not committed** to Git (see root `.gitignore`). Download or export them before running the vision API.

## Required for board detection

| Model | Path | Source |
|-------|------|--------|
| YOLO piece detector | `pretrained/yolov8_chess_pieces.onnx` | Export via `ml/scripts/` or training pipeline |
| Occupancy classifier | `occupancy/occupancy.onnx` | `python -m scripts.setup_pretrained` |
| Piece classifier | `piece_classifier/piece_classifier.onnx` | `python -m scripts.setup_pretrained` |

## Setup (from repo root)

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .
python -m scripts.setup_pretrained
```

## JSON configs

Small `.json` / `.yaml` descriptor files **are** tracked in Git. They reference ONNX paths and preprocessing settings.

## Git LFS (optional)

For teams that want to version large weights:

```bash
git lfs track "ml/models/**/*.onnx"
git lfs track "ml/models/**/*.pt"
```
