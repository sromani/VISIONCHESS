"""Computer-vision pipeline for chess board analysis."""

from vision.board.detector import BoardDetector
from vision.board.exceptions import BoardDetectionError, BoardNotFoundError
from vision.board.grid import BoardGridExtractor
from vision.board.types import BoardDetectionResult, BoardGridResult, Point2D, SquareCrop
from vision.chessboard_detector import ChessboardDetectionResult, ChessboardDetector
from vision.fen import FenBuilder, FenBuildResult, FenIssue, SquarePrediction
from vision.pipeline import VisionPipeline, VisionPipelineResult
from vision.scanner import ScanPipeline, ScannerConfig, ScanResult

__all__ = [
    "BoardDetector",
    "BoardGridExtractor",
    "BoardDetectionError",
    "BoardNotFoundError",
    "BoardDetectionResult",
    "BoardGridResult",
    "SquareCrop",
    "Point2D",
    "ChessboardDetector",
    "ChessboardDetectionResult",
    "VisionPipeline",
    "VisionPipelineResult",
    "ScanPipeline",
    "ScannerConfig",
    "ScanResult",
    "FenBuilder",
    "FenBuildResult",
    "FenIssue",
    "SquarePrediction",
]
