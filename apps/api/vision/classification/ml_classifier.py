"""Production ONNX classifier — staged decode, TTA ensemble, chess correction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.types import BoardGridResult, SquareCrop
from vision.classification.chess_correction import apply_chess_constraints
from vision.classification.ensemble import ensemble_logits
from vision.classification.labels import CLASS_NAMES, NUM_CLASSES
from vision.classification.preprocess import preprocess_rgb_for_model
from vision.classification.staged_inference import StagedPrediction, decode_staged
from vision.classification.types import SquareClassification


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class MlClassifierConfig:
    model_path: Path
    image_size: int = 64
    use_tta: bool = True
    empty_threshold: float = 0.52
    min_piece_confidence: float = 0.28


class MlOnnxClassifier:
    """MobileNet/EfficientNet ONNX with multi-stage pipeline."""

    def __init__(self, config: MlClassifierConfig) -> None:
        import onnxruntime as ort

        self._config = config
        self._session = ort.InferenceSession(
            str(config.model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._class_names = self._load_class_names(config.model_path)
        self._image_size = self._load_image_size(config.model_path)

    @property
    def name(self) -> str:
        return "ml_onnx"

    def classify_squares(
        self,
        squares: list[SquareCrop],
        *,
        soft: bool = False,
    ) -> list[SquareClassification]:
        """Classify squares; ``soft=True`` keeps staged hypotheses without empty demotion."""
        results: list[SquareClassification] = []
        for sq in squares:
            pred = self._classify_square(sq.image, soft=soft)
            results.append(
                SquareClassification(
                    row=sq.row,
                    col=sq.col,
                    square_name=sq.square_name,
                    label=pred.label,
                    confidence=pred.confidence.probability,
                    occupied=False if soft else pred.occupied,
                    occupancy_score=pred.occupancy_prob,
                    empty_reason=None if soft or pred.occupied else "staged_empty",
                )
            )
        return results

    def classify_grid(self, grid: BoardGridResult) -> list[SquareClassification]:
        squares: list[SquareCrop] = list(grid.flat)
        results = self.classify_squares(squares)
        return apply_chess_constraints(results)

    def _classify_square(self, crop_bgr: NDArray[np.uint8], *, soft: bool = False) -> StagedPrediction:
        rgb = _to_rgb(crop_bgr)

        def infer_fn(view: NDArray[np.uint8]) -> NDArray[np.float32]:
            batch = self._preprocess(view)
            return self._session.run([self._output_name], {self._input_name: batch})[0][0]

        if self._config.use_tta:
            logits = ensemble_logits(rgb, infer_fn)
        else:
            logits = infer_fn(rgb)

        pred = decode_staged(logits, empty_threshold=self._config.empty_threshold)
        if soft:
            return pred
        if pred.occupied and pred.confidence.probability < self._config.min_piece_confidence:
            return StagedPrediction(
                label="empty",
                occupied=False,
                color=None,
                piece_kind=None,
                occupancy_prob=1.0 - pred.confidence.probability,
                color_prob=0.0,
                piece_prob=0.0,
                confidence=pred.confidence,
            )
        return pred

    def _preprocess(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        sized = preprocess_rgb_for_model(rgb, size=self._image_size)
        arr = sized.astype(np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        chw = (chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return chw[np.newaxis, ...]

    @staticmethod
    def _load_class_names(model_path: Path) -> list[str]:
        meta = model_path.with_suffix(".json")
        if meta.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            return list(data.get("class_names", CLASS_NAMES))
        return list(CLASS_NAMES)

    @staticmethod
    def _load_image_size(model_path: Path) -> int:
        meta = model_path.with_suffix(".json")
        if meta.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            return int(data.get("image_size", 64))
        return 64


def _to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
