"""Bootstrap LiveChess2FEN imports without installing TensorFlow/Keras."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
LC2FEN_ROOT = _REPO_ROOT / "ml" / "vendor" / "LiveChess2FEN"
PIECE_MODEL_PATH = _REPO_ROOT / "ml" / "models" / "lc2fen" / "MobileNetV2_0p5_all.onnx"
LAPS_MODEL_PATH = LC2FEN_ROOT / "lc2fen" / "detectboard" / "models" / "laps_model.onnx"

_RUNTIME_LOCK = threading.Lock()
_BOOTSTRAPPED = False


def mobilenet_v2_preprocess(x: np.ndarray) -> np.ndarray:
    """Keras MobileNetV2 preprocess_input for float32 RGB in [0, 255]."""
    out = x.astype(np.float32)
    out /= 127.5
    out -= 1.0
    return out


def _load_img_array(img_path: str, target_size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def _install_keras_shim() -> None:
    if "keras" in sys.modules:
        return

    image_utils = ModuleType("keras.utils.image_utils")

    def load_img(img_path: str, target_size: tuple[int, int] | None = None):
        if target_size is None:
            raise ValueError("target_size is required")
        return _load_img_array(img_path, target_size)

    def img_to_array(img, data_format: str = "channels_last"):
        del data_format
        return img

    image_utils.load_img = load_img
    image_utils.img_to_array = img_to_array

    utils = ModuleType("keras.utils")
    utils.image_utils = image_utils

    applications = ModuleType("keras.applications")
    mobilenet_v2 = ModuleType("keras.applications.mobilenet_v2")
    mobilenet_v2.preprocess_input = mobilenet_v2_preprocess
    imagenet_utils = ModuleType("keras.applications.imagenet_utils")
    imagenet_utils.preprocess_input = mobilenet_v2_preprocess
    applications.mobilenet_v2 = mobilenet_v2
    applications.imagenet_utils = imagenet_utils

    models = ModuleType("keras.models")

    def load_model(*_args, **_kwargs):
        raise RuntimeError("Keras inference is disabled; use ONNX Runtime")

    models.load_model = load_model

    keras = ModuleType("keras")
    keras.utils = utils
    keras.models = models
    keras.applications = applications

    sys.modules["keras"] = keras
    sys.modules["keras.utils"] = utils
    sys.modules["keras.utils.image_utils"] = image_utils
    sys.modules["keras.models"] = models
    sys.modules["keras.applications"] = applications
    sys.modules["keras.applications.mobilenet_v2"] = mobilenet_v2
    sys.modules["keras.applications.imagenet_utils"] = imagenet_utils


def _ensure_bootstrapped() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    if not LC2FEN_ROOT.is_dir():
        raise FileNotFoundError(
            f"LiveChess2FEN vendor not found at {LC2FEN_ROOT}. "
            "Run: git clone https://github.com/davidmallasen/LiveChess2FEN.git ml/vendor/LiveChess2FEN"
        )
    if not PIECE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"LC2FEN piece model missing at {PIECE_MODEL_PATH}. "
            "Run: python ml/scripts/setup_lc2fen.py"
        )
    if not LAPS_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"LAPS board model missing at {LAPS_MODEL_PATH}. "
            "Run: python ml/scripts/setup_lc2fen.py"
        )
    _install_keras_shim()
    root = str(LC2FEN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _BOOTSTRAPPED = True


@contextmanager
def lc2fen_runtime():
    """Serialize LC2FEN calls and use vendor-relative model paths."""
    _ensure_bootstrapped()
    previous_cwd = os.getcwd()
    os.chdir(LC2FEN_ROOT)
    with _RUNTIME_LOCK:
        try:
            yield
        finally:
            os.chdir(previous_cwd)
