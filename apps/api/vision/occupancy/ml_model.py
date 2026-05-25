"""Optional binary ML occupancy model (empty vs occupied)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.inference.ml_debug_types import OccupancyMlDebug
from vision.inference.model_registry import ModelArtifact, resolve_occupancy_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class MlOccupancyConfig:
    model_path: Path
    image_size: int = 64
    input_name: str = "square"
    output_name: str = "logits"


class MlOccupancyModel:
    """ONNX binary classifier — occupied probability."""

    def __init__(self, config: MlOccupancyConfig) -> None:
        import onnxruntime as ort

        self._config = config
        self._session = ort.InferenceSession(
            str(config.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = config.input_name
        self._output_name = config.output_name

    @classmethod
    def from_artifact(cls, artifact: ModelArtifact) -> MlOccupancyModel:
        return cls(
            MlOccupancyConfig(
                model_path=artifact.path,
                image_size=artifact.image_size,
                input_name=artifact.input_name,
                output_name=artifact.output_name,
            )
        )

    def predict(self, crop_bgr: NDArray[np.uint8]) -> float:
        return self.predict_debug(crop_bgr).occupied_probability

    def predict_debug(self, crop_bgr: NDArray[np.uint8]) -> OccupancyMlDebug:
        visual, chw = self._preprocess_pair(crop_bgr)
        logits = self._session.run([self._output_name], {self._input_name: chw})[0][0]
        logits = logits.astype(np.float32)
        if logits.size == 1:
            occupied = float(_sigmoid(float(logits[0])))
            logit_pair = (float(-logits[0]), float(logits[0]))
        else:
            probs = _softmax(logits)
            occupied = float(probs[1]) if probs.size > 1 else float(probs[0])
            logit_pair = tuple(float(x) for x in logits[:2])
        return OccupancyMlDebug(
            occupied_probability=occupied,
            empty_probability=float(1.0 - occupied),
            logits=logit_pair,
            model_input_size=self._config.image_size,
            onnx_input_bgr=visual,
        )

    def _preprocess_pair(self, crop_bgr: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        size = self._config.image_size
        if rgb.shape[0] != size or rgb.shape[1] != size:
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        visual = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        arr = rgb.astype(np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        chw = (chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return visual, chw[np.newaxis, ...]

    def _preprocess(self, crop_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        _, chw = self._preprocess_pair(crop_bgr)
        return chw


def resolve_occupancy_model_path(explicit: Path | None = None) -> Path | None:
    artifact = resolve_occupancy_model(explicit)
    return artifact.path if artifact is not None else None


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32)
