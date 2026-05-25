"""FastAPI dependencies."""

from functools import lru_cache

from app.core.settings import get_settings
from app.services.analysis_service import AnalysisService
from app.services.scan_service import ScanRepository, ScanService
from app.utils.file_storage import StorageService
from vision.board.split import BoardSquareSplitter
from vision.classification.pipeline import ClassificationPipeline, ClassificationPipelineConfig
from vision.pipeline import VisionPipeline
from vision.lc2fen.adapter import LC2FENAdapter
from vision.scanner import PieceDetectionPipeline, ScanPipeline, ScannerConfig


@lru_cache
def get_scan_pipeline() -> ScanPipeline:
    settings = get_settings()
    return ScanPipeline(
        ScannerConfig.from_settings(
            output_size=settings.board_output_size,
            margin_ratio=settings.classification_margin_ratio,
            model_path=settings.piece_classifier_path,
            stockfish_path=settings.stockfish_path,
            use_stockfish=settings.classification_use_stockfish_tiebreak,
        )
    )


@lru_cache
def get_lc2fen_adapter() -> LC2FENAdapter:
    return LC2FENAdapter()


@lru_cache
def get_piece_detection_pipeline() -> PieceDetectionPipeline:
    settings = get_settings()
    return PieceDetectionPipeline(
        ScannerConfig.piece_detection_only(
            output_size=settings.board_output_size,
            margin_ratio=settings.classification_margin_ratio,
            model_path=settings.piece_classifier_path,
        )
    )


@lru_cache
def get_chessboard_detector():
    """Legacy alias — prefer get_scan_pipeline()."""
    from vision.chessboard_detector import ChessboardDetector

    return ChessboardDetector()


@lru_cache
def get_classification_pipeline() -> ClassificationPipeline:
    settings = get_settings()
    return ClassificationPipeline(
        ClassificationPipelineConfig(
            margin_ratio=settings.classification_margin_ratio,
            model_path=settings.piece_classifier_path,
            stockfish_path=settings.stockfish_path,
            use_stockfish_tiebreak=settings.classification_use_stockfish_tiebreak,
        )
    )


@lru_cache
def get_square_splitter() -> BoardSquareSplitter:
    return BoardSquareSplitter()


@lru_cache
def get_storage_service() -> StorageService:
    return StorageService(get_settings())


@lru_cache
def get_scan_repository() -> ScanRepository:
    return ScanRepository()


@lru_cache
def get_vision_pipeline() -> VisionPipeline:
    return VisionPipeline(scan_pipeline=get_scan_pipeline())


def get_scan_service() -> ScanService:
    return ScanService(
        settings=get_settings(),
        storage=get_storage_service(),
        repository=get_scan_repository(),
        pipeline=get_vision_pipeline(),
        scan_pipeline=get_scan_pipeline(),
    )


def get_analysis_service() -> AnalysisService:
    return AnalysisService(get_settings())
