"""Single-square and board-level inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms import Compose

from training.export import load_checkpoint
from training.labels import CLASS_NAMES, IDX_TO_CLASS, NUM_CLASSES
from training.model import build_model
from training.transforms import inference_transform

if TYPE_CHECKING:
    from training.config import TrainConfig


@dataclass(frozen=True, slots=True)
class PiecePrediction:
    label: str
    confidence: float
    class_index: int


class PieceClassifier:
    """Fast batched inference for 64×64 square crops."""

    def __init__(
        self,
        model: nn.Module,
        transform: Compose,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.transform = transform

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | str) -> PieceClassifier:
        payload = load_checkpoint(Path(checkpoint_path))
        backbone = payload["backbone"]
        num_classes = payload.get("num_classes", NUM_CLASSES)
        image_size = payload.get("image_size", 64)

        model = build_model(backbone, num_classes, pretrained=False)
        model.load_state_dict(payload["model_state_dict"])
        transform = inference_transform(image_size)
        return cls(model=model, transform=transform)


    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image] | NDArray[np.uint8]) -> list[PiecePrediction]:
        tensors = self._to_batch(images)
        logits = self.model(tensors)
        probs = torch.softmax(logits, dim=1)
        confidences, indices = probs.max(dim=1)

        results: list[PiecePrediction] = []
        for idx, conf in zip(indices.tolist(), confidences.tolist(), strict=True):
            results.append(
                PiecePrediction(
                    label=IDX_TO_CLASS[idx],
                    confidence=float(conf),
                    class_index=idx,
                )
            )
        return results

    def predict(self, image: Image.Image | NDArray[np.uint8]) -> PiecePrediction:
        return self.predict_batch([image])[0]

    def _to_batch(self, images: list[Image.Image] | NDArray[np.uint8]) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [images]
            else:
                images = [images[i] for i in range(images.shape[0])]

        batch = torch.stack([self.transform(Image.fromarray(img) if isinstance(img, np.ndarray) else img.convert("RGB")) for img in images])
        return batch.to(self.device, non_blocking=True)


class OnnxPieceClassifier:
    """ONNX Runtime backend for minimal latency in production."""

    def __init__(self, onnx_path: Path | str) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._transform = inference_transform(self._input_size(onnx_path))

    def _input_size(self, onnx_path: Path | str) -> int:
        meta_path = Path(onnx_path).with_suffix(".json")
        if meta_path.exists():
            import json

            return int(json.loads(meta_path.read_text(encoding="utf-8"))["image_size"])
        return 64

    def predict_batch(self, images: list[Image.Image] | NDArray[np.uint8]) -> list[PiecePrediction]:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        elif isinstance(images, np.ndarray):
            images = [images[i] for i in range(images.shape[0])]

        tensors = [
            self._transform(img if isinstance(img, Image.Image) else Image.fromarray(img)).numpy()
            for img in images
        ]
        batch = np.stack(tensors, axis=0).astype(np.float32)
        logits = self._session.run([self._output_name], {self._input_name: batch})[0]

        results: list[PiecePrediction] = []
        for row in logits:
            idx = int(row.argmax())
            exp = np.exp(row - row.max())
            conf = float(exp[idx] / exp.sum())
            results.append(PiecePrediction(label=IDX_TO_CLASS[idx], confidence=conf, class_index=idx))
        return results

    def predict(self, image: Image.Image | NDArray[np.uint8]) -> PiecePrediction:
        return self.predict_batch([image])[0]


def classify_board_squares(
    warped_board: NDArray[np.uint8],
    classifier: PieceClassifier | OnnxPieceClassifier,
    board_size: int = 800,
) -> list[list[PiecePrediction]]:
    """Split a top-down board into 64 squares and classify each."""
    square_size = board_size // 8
    margin = int(square_size * 0.08)
    grid: list[list[PiecePrediction]] = []

    crops: list[NDArray[np.uint8]] = []
    positions: list[tuple[int, int]] = []

    for row in range(8):
        for col in range(8):
            y0 = row * square_size + margin
            x0 = col * square_size + margin
            y1 = (row + 1) * square_size - margin
            x1 = (col + 1) * square_size - margin
            crop = warped_board[y0:y1, x0:x1]
            # OpenCV boards are BGR; classifier expects RGB.
            crops.append(crop[:, :, ::-1].copy())
            positions.append((row, col))

    preds = classifier.predict_batch(crops)

    grid = [[PiecePrediction("empty", 0.0, 0) for _ in range(8)] for _ in range(8)]
    for (row, col), pred in zip(positions, preds, strict=True):
        grid[row][col] = pred
    return grid


def piece_map_to_fen_placement(grid: list[list[PiecePrediction]]) -> str:
    """Convert predictions to FEN piece placement (board only)."""
    fen_map = {
        "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
        "white_rook": "R", "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
        "black_rook": "r", "black_queen": "q", "black_king": "k",
    }

    ranks: list[str] = []
    for row in grid:
        rank = ""
        empty = 0
        for pred in row:
            if pred.label == "empty":
                empty += 1
            else:
                if empty:
                    rank += str(empty)
                    empty = 0
                rank += fen_map[pred.label]
        if empty:
            rank += str(empty)
        ranks.append(rank)
    return "/".join(ranks)
