"""High-level chess board detector — grid solve + mesh rectification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vision.board.config import BoardDetectorConfig
from vision.board.exceptions import BoardNotFoundError
from vision.board.io import decode_image_bytes
from vision.board.types import BoardDetectionResult
from vision.chessboard_detector import ChessboardDetector, ChessboardDetectorConfig


class BoardDetector:
    """Detect a chess board via full 9×9 grid solve and piecewise mesh rectification."""

    def __init__(self, config: BoardDetectorConfig | None = None) -> None:
        cfg = config or BoardDetectorConfig()
        self._config = cfg
        self._detector = ChessboardDetector(
            ChessboardDetectorConfig(
                output_size=cfg.output_size,
                max_detection_dim=cfg.max_detection_dim,
                max_aspect_ratio_deviation=cfg.max_aspect_ratio_deviation,
                min_cosine_angle=cfg.min_cosine_angle,
                min_score=cfg.min_score,
            )
        )

    @property
    def config(self) -> BoardDetectorConfig:
        return self._config

    def detect(self, image: NDArray[np.uint8]) -> BoardDetectionResult:
        if image.size == 0:
            raise BoardNotFoundError("Empty image")

        result = self._detector.detect(image, collect_debug=False)
        return BoardDetectionResult(
            corners=result.corners,
            homography=result.homography,
            warped_board=result.warped_image,
            confidence=result.confidence,
            original_size=(result.original_width, result.original_height),
            output_size=result.output_width,
        )

    def detect_from_bytes(self, data: bytes) -> BoardDetectionResult:
        return self.detect(decode_image_bytes(data))
