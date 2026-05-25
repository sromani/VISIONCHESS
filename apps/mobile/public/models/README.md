# Vision models (not in git)

Large ONNX files are copied on install:

```bash
npm run copy:vision-assets --prefix apps/mobile
```

Requires `ml/models/pretrained/yolov8_chess_pieces.onnx` and LC2FEN `laps_model.onnx` (see `ml/scripts/`).
