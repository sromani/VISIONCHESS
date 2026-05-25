# Chess Vision Architecture — Production Integration Plan

VisionChess target: **image → valid FEN**, comparable to Chessvision.ai / Lichess scanner / Chess.com vision.

Geometry (modules A–C) is largely working. This document defines how to integrate proven open-source projects for **occupancy, classification, and FEN validation** (modules D–G).

---

## 1. Open-source repos — what to reuse

### [chesscog](https://github.com/georg-wolflein/chesscog) (★138, J. Imaging 2021)

**What it does:** End-to-end research pipeline — board localisation → occupancy CNN → piece CNN → FEN post-processing. Trained on ~5k synthetic 3D renders. Published error rate ~0.23% per square on test set.

| Component | Reuse in VisionChess | Integration |
|-----------|---------------------|-------------|
| Board localisation | **Optional fallback** | Our Hough/mesh is already strong; chesscog corner detector useful when grid lines fail |
| Occupancy classifier | **Yes — primary reference** | Separate binary CNN per square (ResNet/AlexNet, 100×100). Matches our staged design |
| Piece classifier | **Yes — architecture + weights** | 12-way CNN on occupied squares only; ImageNet pretrain + fine-tune |
| Training scripts | **Yes** | `occupancy_classifier/train`, `piece_classifier/train`, dataset creation |
| Pretrained models | **Download & adapt** | `python -m chesscog.occupancy_classifier.download_model` |

**Do not fork wholesale:** chesscog is GPL, uses different grid math, and localisation is weaker than our mesh solver on skewed photos. **Extract ML patterns, not the full repo.**

---

### [LiveChess2FEN](https://github.com/davidmallasen/LiveChess2FEN) (★92, arXiv 2020)

**What it does:** Real-time OTB digitization on Jetson Nano. Iterative corner refinement (`laps.py`) + CNN piece classification (~55k labeled images). ONNX/TensorRT inference.

| Component | Reuse | Integration |
|-----------|-------|-------------|
| Board detection (`detectboard/laps.py`) | **Reference only** | Iterative corner refinement when our localization confidence is low |
| Piece CNN + ONNX export | **Yes** | Pretrained ONNX as optional backend; compare against our MobileNet training |
| 13-class direct classification | **Alternative path** | Single CNN without separate occupancy (simpler but less accurate per chesscog) |

**Best for:** ONNX runtime patterns, embedded deployment, piece classifier baselines.

---

### [ChessboardDetect](https://github.com/Elucidation/ChessboardDetect) (★124)

**What it does:** Collection of classical board detection algorithms on real match photos (Hough, contour, template variants).

| Component | Reuse | Integration |
|-----------|-------|-------------|
| Algorithm benchmarks | **Yes** | Regression tests on hard photos when localization fails |
| Specific detectors | **Fallback backends** | Plug into `vision/backends/localization/` when primary Hough fails |

**Not ML** — useful as localization ensemble, not for piece ID.

---

### [python-chess](https://github.com/niklasf/python-chess)

**What it does:** Chess rules, FEN parsing, legality, move generation.

| Component | Reuse | Integration |
|-----------|-------|-------------|
| `chess.Board(fen).is_valid()` | **Already used** | Expand in `vision/validation/` |
| Piece count / pawn rank checks | **Already used** | Centralize in `BoardValidator` |
| FEN repair | **Enhance** | Use board.legal_moves for impossible positions |

---

### [Stockfish](https://github.com/official-stockfish/Stockfish)

**What it does:** Engine evaluation for position plausibility.

| Component | Reuse | Integration |
|-----------|-------|-------------|
| `engine.analyse()` depth-8 | **Already optional** | Make first-class in hypothesis scoring (module G) |
| Multi-PV | **Add** | Compare top 3 FEN candidates by eval consistency |

---

## 2. Target module architecture (A–G)

```
apps/api/vision/
├── board/              # A. BOARD LOCALIZATION (keep — working)
├── rectification/      # B. RECTIFICATION (mesh_rectify stage → move here)
├── extraction/         # C. SQUARE EXTRACTION (upscale + crop_quality)
├── occupancy/          # D. OCCUPANCY (ML-only path, soft probabilities)
├── classification/     # E. PIECE CLASSIFIER (ONNX required in production)
├── hypotheses/         # NEW — board hypothesis generation
├── validation/         # F. python-chess + G. Stockfish scoring
├── inference/          # Unified ONNX runtime (occupancy + piece)
├── backends/           # Optional: chesscog, lc2fen adapters
├── dataset/            # Auto dataset builder
└── scanner/            # Orchestration only
```

### Pipeline flow

```
image
  → A localization (Hough + mesh)
  → B rectification (piecewise warp, 2048 upscale)
  → C extraction (64 high-res enhanced crops)
  → D occupancy probabilities (ONNX binary, no hard threshold)
  → E piece probabilities (ONNX 13-class, soft mode)
  → hypotheses (joint occ × piece, orientation variants, top-k alts)
  → F validation (python-chess legality, piece counts)
  → G Stockfish plausibility
  → best FEN
```

---

## 3. What to eliminate

| Remove | Replace with |
|--------|--------------|
| Heuristic piece classifier (Otsu/contours) | ONNX 13-class (required in prod) |
| Occupancy edge/variance/entropy fusion | ONNX binary occupancy |
| Early `occupied=True/False` | Soft probs until validation |
| Silhouette/template empty model | ML occupancy trained on real empty squares |
| Single-pass FEN | Hypothesis engine + global scoring |

Heuristic backends remain as `dev_fallback` only when `require_ml_models=false`.

---

## 4. ML training (`ml/`)

Already present; align with chesscog:

| Model | Input | Classes | Data source |
|-------|-------|---------|-------------|
| Occupancy | 64×64 or 128×128 RGB | 2 (empty/occupied) | Lichess puzzles + scan exports + synthetic |
| Piece | 64×64 RGB | 13 (empty + 12 pieces) | Same + chesscog-style 3D renders |

Commands:
```bash
cd ml
python scripts/build_from_lichess.py
python scripts/build_occupancy_dataset.py
python -m training.occupancy_cli --epochs 30
python -m training.cli --epochs 30
```

Ship models to `ml/models/occupancy/occupancy.onnx` and `ml/models/piece_classifier/best.onnx`.

---

## 5. Dataset builder

`vision/dataset/builder.py` saves:
```
dataset/
  train/
    empty/
    white_pawn/
    ...
  manifest.jsonl      # path, label, confidence, job_id, square
  corrections.jsonl   # human relabels from review UI
  occupancy/          # binary subset for occupancy retrain
```

Enhanced: save high-res `analysis/` crops alongside 64px export.

---

## 6. Debug UI tabs

| Tab | Backend key |
|-----|-------------|
| Original | `original` |
| Detected lines | `detected_lines` |
| Intersections | `intersections` |
| Mesh | `mesh` |
| Rectified board | `rectified_board` |
| Rectified upscaled | `rectified_upscaled` |
| Square extraction | `square_extraction` |
| Crop quality | `crop_quality` |
| Occupancy heatmap | `occupancy` |
| Occupancy signals | `occupancy_detail` |
| Classifier heatmap | `classifier_confidence` |
| FEN candidates | `fen_candidates` |
| Final board | `final_board` |

---

## 7. Integration roadmap

## ML integration (piece recognition focus)

### Bootstrap local model (done if `piece_classifier.onnx` exists)
```bash
cd ml
python scripts/setup_pretrained.py --source synthetic --epochs 20
```

### Import pretrained external models
```bash
# chesscog (ResNet, ~0.23% error on synthetic benchmark)
pip install chesscog
python scripts/setup_pretrained.py --source chesscog

# LiveChess2FEN — download .onnx from GitHub releases first
python scripts/setup_pretrained.py --source lc2fen --lc2fen-dir PATH/to/LiveChess2FEN/data/models

# Any 13-class ONNX + metadata JSON
python scripts/setup_pretrained.py --source import-onnx --onnx-path model.onnx --meta-path model.json
```

### Import external datasets
```bash
python scripts/import_class_folder.py /path/to/dataset --out data/squares/train
python -m training.cli --epochs 30
```

### Model search paths (`vision/inference/model_registry.py`)
1. `ml/models/piece_classifier/best.onnx`
2. `ml/models/piece_classifier/piece_classifier.onnx`
3. `ml/models/pretrained/chesscog_piece.onnx`
4. `ml/models/pretrained/lc2fen_piece.onnx`

### Inference pipeline
- `PieceInferencePipeline` — batched 64-square ONNX, **no heuristics**
- Classification stage uses ML only; heuristics only if `allow_heuristic_fallback=True`

- [x] Soft occupancy → validation finalize
- [x] High-res crop pipeline before ML
- [x] Hypothesis engine + Stockfish scoring module
- [x] ML-only config flag
- [x] Debug UI/API parity

### Phase 2
- [ ] Train & ship piece_classifier.onnx on Lichess + real scans
- [ ] Retrain occupancy on mixed real/synthetic
- [ ] chesscog weight import adapter (optional)
- [ ] LiveChess2FEN ONNX baseline comparison

### Phase 3
- [ ] Learned keypoint detector (replace Hough fallback)
- [ ] Real-photo benchmark suite (100+ labeled positions)
- [ ] Active learning loop from dataset review UI

---

## 8. Success criteria

- Valid FEN on skewed phone photos with standard Staunton sets
- Stable across chess.com / lichess themes
- Occupancy FPR < 5% on empty squares
- Piece accuracy > 90% per square on benchmark set
- End-to-end latency < 10s on CPU, < 3s on GPU
