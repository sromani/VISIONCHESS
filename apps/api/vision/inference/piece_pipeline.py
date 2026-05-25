"""ML-only batched piece classification — no heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.board.types import BoardGridResult, SquareCrop
from vision.classification.backend_protocol import PieceClassifierBackend
from vision.classification.labels import CLASS_NAMES
from vision.classification.preprocess import preprocess_rgb_for_model
from vision.classification.types import SquareClassification
from vision.inference.ml_debug_types import ClassPrediction, PieceMlDebug
from vision.inference.model_registry import ModelArtifact, resolve_piece_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class PieceInferenceConfig:
    model_path: Path | None = None
    batch_size: int = 64
    use_tta: bool = False


class PieceInferencePipeline(PieceClassifierBackend):
    """Production piece classifier — single ONNX, batched 64-square inference."""

    def __init__(self, artifact: ModelArtifact, config: PieceInferenceConfig | None = None) -> None:
        import onnxruntime as ort

        self._artifact = artifact
        self._config = config or PieceInferenceConfig()
        self._session = ort.InferenceSession(
            str(artifact.path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = artifact.input_name
        self._output_name = artifact.output_name
        self._class_names = list(artifact.class_names) if artifact.class_names else list(CLASS_NAMES)
        self._image_size = artifact.image_size

    @property
    def name(self) -> str:
        return f"ml_onnx_{self._artifact.source}"

    @classmethod
    def from_registry(cls, config: PieceInferenceConfig | None = None) -> PieceInferencePipeline:
        cfg = config or PieceInferenceConfig()
        artifact = resolve_piece_model(cfg.model_path)
        if artifact is None:
            msg = (
                "No piece classifier ONNX found. Run: "
                "cd ml && python scripts/setup_pretrained.py --source synthetic"
            )
            raise RuntimeError(msg)
        return cls(artifact, cfg)

    def classify_squares(
        self,
        squares: list[SquareCrop],
        *,
        soft: bool = True,
        capture_debug: bool = False,
    ) -> list[SquareClassification]:
        if not squares:
            return []

        logits_all, debug_rows = self._run_batches(squares, capture_debug=capture_debug)
        if capture_debug:
            self._last_piece_debug = debug_rows

        results: list[SquareClassification] = []
        for sq, row_logits in zip(squares, logits_all, strict=True):
            idx = int(row_logits.argmax())
            probs = _softmax(row_logits)
            conf = float(probs[idx])
            label = self._artifact.label_for_index(idx)
            results.append(
                SquareClassification(
                    row=sq.row,
                    col=sq.col,
                    square_name=sq.square_name,
                    label=label,
                    confidence=conf,
                    occupied=False,
                    occupancy_score=conf if label != "empty" else 1.0 - conf,
                    empty_reason=None,
                )
            )
        return results

    def classify_squares_with_debug(
        self,
        squares: list[SquareCrop],
    ) -> tuple[list[SquareClassification], list[PieceMlDebug]]:
        preds = self.classify_squares(squares, soft=True, capture_debug=True)
        return preds, list(getattr(self, "_last_piece_debug", []))

    def consume_piece_debug(self) -> list[PieceMlDebug]:
        rows = list(getattr(self, "_last_piece_debug", []))
        self._last_piece_debug = []
        return rows

    @property
    def artifact(self) -> ModelArtifact:
        return self._artifact

    def classify_grid(self, grid: BoardGridResult) -> list[SquareClassification]:
        return self.classify_squares(list(grid.flat), soft=True)

    def classify_crops(
        self,
        crops: list[NDArray[np.uint8]],
    ) -> list[tuple[str, float, tuple[ClassPrediction, ...]]]:
        """Classify arbitrary BGR crops — returns (label, confidence, top3) per crop."""
        if not crops:
            return []

        logits_all, _ = self._run_batches_from_images(crops, capture_debug=False)
        results: list[tuple[str, float, tuple[ClassPrediction, ...]]] = []
        for row_logits in logits_all:
            idx = int(row_logits.argmax())
            probs = _softmax(row_logits)
            conf = float(probs[idx])
            label = self._artifact.label_for_index(idx)
            ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            top3 = tuple(
                ClassPrediction(
                    label=self._artifact.label_for_index(i),
                    probability=float(probs[i]),
                    logit=float(row_logits[i]),
                )
                for i in ranked[:3]
            )
            results.append((label, conf, top3))
        return results

    def classify_crops_with_debug(
        self,
        crops: list[NDArray[np.uint8]],
    ) -> tuple[list[tuple[str, float, tuple[ClassPrediction, ...]]], list[PieceMlDebug]]:
        logits_all, debug_rows = self._run_batches_from_images(crops, capture_debug=True)
        results: list[tuple[str, float, tuple[ClassPrediction, ...]]] = []
        for row_logits in logits_all:
            idx = int(row_logits.argmax())
            probs = _softmax(row_logits)
            conf = float(probs[idx])
            label = self._artifact.label_for_index(idx)
            ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            top3 = tuple(
                ClassPrediction(
                    label=self._artifact.label_for_index(i),
                    probability=float(probs[i]),
                    logit=float(row_logits[i]),
                )
                for i in ranked[:3]
            )
            results.append((label, conf, top3))
        return results, debug_rows

    def _run_batches_from_images(
        self,
        images: list[NDArray[np.uint8]],
        *,
        capture_debug: bool,
    ) -> tuple[list[NDArray[np.float32]], list[PieceMlDebug]]:
        bs = max(1, self._config.batch_size)
        all_logits: list[NDArray[np.float32]] = []
        debug_rows: list[PieceMlDebug] = []
        for start in range(0, len(images), bs):
            chunk = images[start : start + bs]
            batch_tensors: list[NDArray[np.float32]] = []
            visuals: list[NDArray[np.uint8]] = []
            for img in chunk:
                visual, chw = self._preprocess_pair(img)
                batch_tensors.append(chw)
                visuals.append(visual)
            batch = np.stack(batch_tensors, axis=0).astype(np.float32)
            logits = self._session.run([self._output_name], {self._input_name: batch})[0]
            for row, visual in zip(logits, visuals, strict=True):
                row_f = row.astype(np.float32)
                all_logits.append(row_f)
                if capture_debug:
                    debug_rows.append(self._build_piece_debug(row_f, visual))
        return all_logits, debug_rows

    def _run_batches(
        self,
        squares: list[SquareCrop],
        *,
        capture_debug: bool = False,
    ) -> tuple[list[NDArray[np.float32]], list[PieceMlDebug]]:
        bs = max(1, self._config.batch_size)
        all_logits: list[NDArray[np.float32]] = []
        debug_rows: list[PieceMlDebug] = []
        for start in range(0, len(squares), bs):
            chunk = squares[start : start + bs]
            batch_tensors: list[NDArray[np.float32]] = []
            visuals: list[NDArray[np.uint8]] = []
            for sq in chunk:
                visual, chw = self._preprocess_pair(sq.image)
                batch_tensors.append(chw)
                visuals.append(visual)
            batch = np.stack(batch_tensors, axis=0).astype(np.float32)
            logits = self._session.run([self._output_name], {self._input_name: batch})[0]
            for row, visual in zip(logits, visuals, strict=True):
                row_f = row.astype(np.float32)
                all_logits.append(row_f)
                if capture_debug:
                    debug_rows.append(self._build_piece_debug(row_f, visual))
        return all_logits, debug_rows

    def _build_piece_debug(self, logits: NDArray[np.float32], visual: NDArray[np.uint8]) -> PieceMlDebug:
        probs = _softmax(logits)
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top3 = tuple(
            ClassPrediction(
                label=self._artifact.label_for_index(i),
                probability=float(probs[i]),
                logit=float(logits[i]),
            )
            for i in ranked[:3]
        )
        return PieceMlDebug(
            top3=top3,
            logits=tuple(float(x) for x in logits),
            class_names=tuple(self._class_names),
            model_input_size=self._image_size,
            onnx_input_bgr=visual,
        )

    def _preprocess(self, crop_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        _, chw = self._preprocess_pair(crop_bgr)
        return chw

    def _preprocess_pair(self, crop_bgr: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr.ndim == 3 else crop_bgr
        sized = preprocess_rgb_for_model(rgb, size=self._image_size)
        visual = cv2.cvtColor(sized, cv2.COLOR_RGB2BGR)
        arr = sized.astype(np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        chw = (chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return visual, chw


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32)
