"""LiveChess2FEN adapter — image bytes to FEN via vendored pipeline."""

from __future__ import annotations

import glob
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from vision.lc2fen.common import SquarePrediction, draw_corner_overlay, find_rectified_image, warp_preview
from vision.lc2fen.bootstrap import (
    LC2FEN_ROOT,
    PIECE_MODEL_PATH,
    lc2fen_runtime,
    mobilenet_v2_preprocess,
)
from vision.lc2fen.fen_validate import FenValidation, validate_placement_fen
from vision.lc2fen.geometry import rectify_board_from_bytes

IMG_SIZE = 224
A1_POS_DEFAULT = "BL"

_IDX_TO_PIECE = {
    0: "B",
    1: "K",
    2: "N",
    3: "P",
    4: "Q",
    5: "R",
    6: "empty",
    7: "b",
    8: "k",
    9: "n",
    10: "p",
    11: "q",
    12: "r",
}


@dataclass
class LC2FENResult:
    fen: str
    corners: list[list[int]]
    original_width: int
    original_height: int
    warped_bgr: np.ndarray
    rectified_bgr: np.ndarray
    square_predictions: list[SquarePrediction]
    validation: FenValidation
    a1_pos: str
    processing_ms: int
    metadata: dict


class LC2FENAdapter:
    """Wrap vendored LiveChess2FEN predict_board with ONNX Runtime (no Keras)."""

    def __init__(self, *, a1_pos: str = A1_POS_DEFAULT) -> None:
        self.a1_pos = a1_pos
        self._session: onnxruntime.InferenceSession | None = None

    def _get_session(self) -> onnxruntime.InferenceSession:
        if self._session is None:
            self._session = onnxruntime.InferenceSession(str(PIECE_MODEL_PATH))
        return self._session

    def predict_from_path(
        self,
        image_path: Path,
        *,
        previous_fen: str | None = None,
    ) -> LC2FENResult:
        started = time.perf_counter()
        image_path = image_path.resolve()
        work_dir = image_path.parent

        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Unable to read image: {image_path}")

        with lc2fen_runtime():
            from lc2fen.predict_board import predict_board

            sess = self._get_session()
            input_name = sess.get_inputs()[0].name

            def obtain_piece_probs(piece_paths: list[str]) -> list[list[float]]:
                predictions: list[list[float]] = []
                for piece_path in piece_paths:
                    img = cv2.imread(piece_path)
                    if img is None:
                        raise ValueError(f"Unable to read crop: {piece_path}")
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                    tensor = np.expand_dims(rgb.astype(np.float32), axis=0)
                    tensor = mobilenet_v2_preprocess(tensor)
                    predictions.append(sess.run(None, {input_name: tensor})[0][0].tolist())
                return predictions

            fen, corners = predict_board(
                str(image_path),
                self.a1_pos,
                obtain_piece_probs,
                previous_fen=previous_fen,
            )

            tmp_dir = work_dir / "tmp"
            rectified_path = find_rectified_image(tmp_dir, image_path.name)
            rectified_bgr = cv2.imread(str(rectified_path)) if rectified_path else original.copy()
            if rectified_bgr is None:
                rectified_bgr = original.copy()

            piece_probs = obtain_piece_probs(sorted(glob.glob(str(tmp_dir / "pieces" / "*.jpg"))))
            square_predictions = _build_square_predictions(piece_probs, self.a1_pos)

        warped = warp_preview(original, corners)
        validation = validate_placement_fen(fen)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return LC2FENResult(
            fen=fen,
            corners=corners,
            original_width=original.shape[1],
            original_height=original.shape[0],
            warped_bgr=warped,
            rectified_bgr=rectified_bgr,
            square_predictions=square_predictions,
            validation=validation,
            a1_pos=self.a1_pos,
            processing_ms=elapsed_ms,
            metadata={
                "backend": "lc2fen",
                "piece_model": "ml/models/lc2fen/MobileNetV2_0p5_all.onnx",
                "vendor_root": str(LC2FEN_ROOT),
                "img_size": IMG_SIZE,
                "a1_pos": self.a1_pos,
            },
        )

    def predict_yolo_from_bytes(
        self,
        data: bytes,
        *,
        job_dir: Path,
        filename: str = "input.jpg",
        conf_threshold: float = 0.35,
    ):
        """LC2FEN geometry + YOLO piece detection (no 64-square classifier)."""
        import shutil
        import time

        from vision.lc2fen.yolo_pieces import LC2FENYoloResult, run_lc2fen_yolo_pipeline

        started = time.perf_counter()
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            geometry = rectify_board_from_bytes(
                data,
                job_dir=job_dir,
                filename=filename,
                a1_pos=self.a1_pos,
            )
            result = run_lc2fen_yolo_pipeline(geometry, conf_threshold=conf_threshold)
            result.metadata["processing_ms"] = int((time.perf_counter() - started) * 1000)
            result.metadata["a1_pos"] = self.a1_pos
            return result
        finally:
            tmp_dir = job_dir / "tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def predict_from_bytes(
        self,
        data: bytes,
        *,
        job_dir: Path,
        filename: str = "input.jpg",
        previous_fen: str | None = None,
    ) -> LC2FENResult:
        job_dir.mkdir(parents=True, exist_ok=True)
        image_path = job_dir / filename
        image_path.write_bytes(data)
        try:
            return self.predict_from_path(image_path, previous_fen=previous_fen)
        finally:
            tmp_dir = job_dir / "tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)


def _find_rectified_image(tmp_dir: Path, original_name: str) -> Path | None:
    return find_rectified_image(tmp_dir, original_name)


def _warp_preview(image_bgr: np.ndarray, corners: list[list[int]]) -> np.ndarray:
    return warp_preview(image_bgr, corners)


def _square_name_from_indices(row: int, col: int, a1_pos: str) -> tuple[str, int, int]:
    """Map LC2FEN crop indices (top-left origin) to algebraic square + matrix row/col."""
    if a1_pos == "BL":
        rank = 8 - row
        file_idx = col
        matrix_row = row
        matrix_col = col
    elif a1_pos == "BR":
        rank = 8 - row
        file_idx = 7 - col
        matrix_row = row
        matrix_col = 7 - col
    elif a1_pos == "TL":
        rank = row + 1
        file_idx = col
        matrix_row = 7 - row
        matrix_col = col
    elif a1_pos == "TR":
        rank = row + 1
        file_idx = 7 - col
        matrix_row = 7 - row
        matrix_col = 7 - col
    else:
        raise ValueError(f"Unsupported a1_pos: {a1_pos}")

    name = f"{chr(ord('a') + file_idx)}{rank}"
    return name, matrix_row, matrix_col


def _build_square_predictions(
    piece_probs: list[list[float]],
    a1_pos: str,
) -> list[SquarePrediction]:
    if len(piece_probs) != 64:
        raise ValueError(f"Expected 64 square predictions, got {len(piece_probs)}")

    predictions: list[SquarePrediction] = []
    for idx in range(64):
        row = idx // 8
        col = idx % 8
        probs = piece_probs[idx]
        top_idx = int(np.argmax(probs))
        label = _IDX_TO_PIECE[top_idx]
        confidence = float(probs[top_idx])
        name, matrix_row, matrix_col = _square_name_from_indices(row, col, a1_pos)
        predictions.append(
            SquarePrediction(
                name=name,
                row=matrix_row,
                col=matrix_col,
                label=label,
                confidence=confidence,
                occupied=label != "empty",
                probabilities=probs,
            )
        )

    predictions.sort(key=lambda sq: (sq.row, sq.col))
    return predictions


def build_square_montage(rectified_bgr: np.ndarray, squares: list[SquarePrediction]) -> np.ndarray:
    """8×8 montage with predicted labels."""
    board = rectified_bgr.copy()
    h, w = board.shape[:2]
    cell = h // 8
    for sq in squares:
        x0 = sq.col * cell
        y0 = sq.row * cell
        color = (40, 180, 90) if sq.occupied else (80, 80, 80)
        label = sq.label if sq.occupied else "."
        cv2.rectangle(board, (x0, y0), (x0 + cell - 1, y0 + cell - 1), color, 2)
        cv2.putText(
            board,
            f"{label} {sq.confidence:.2f}",
            (x0 + 4, y0 + max(16, cell // 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return board


def fen_to_board_matrix(fen: str, squares: list[SquarePrediction]) -> list[list[dict[str, str | float]]]:
    by_name = {sq.name: sq for sq in squares}
    rows = fen.split("/")
    matrix: list[list[dict[str, str | float]]] = []
    for rank_idx, row in enumerate(rows):
        row_cells: list[dict[str, str | float]] = []
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                for _ in range(int(ch)):
                    name = f"{chr(ord('a') + file_idx)}{8 - rank_idx}"
                    sq = by_name.get(name)
                    row_cells.append(
                        {
                            "label": sq.label if sq else "empty",
                            "confidence": round(sq.confidence, 4) if sq else 0.0,
                        }
                    )
                    file_idx += 1
            else:
                name = f"{chr(ord('a') + file_idx)}{8 - rank_idx}"
                sq = by_name.get(name)
                label = "empty" if ch == "_" else ch
                if sq and sq.label != "empty":
                    label = sq.label
                row_cells.append(
                    {
                        "label": label,
                        "confidence": round(sq.confidence, 4) if sq else 1.0,
                    }
                )
                file_idx += 1
        matrix.append(row_cells)
    return matrix
