"""Scan processing service."""

from __future__ import annotations

import time

import cv2

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.settings import Settings
from app.models.scan import ScanRecord, ScanStatus
from app.utils.file_storage import StorageService
from app.utils.image_validation import ValidatedImage
from vision.board.exceptions import BoardNotFoundError
from vision.pipeline import VisionPipeline
from vision.scanner import ScanPipeline, ScannerConfig

logger = get_logger(__name__)


class ScanRepository:
    """In-memory scan store — replace with PostgreSQL in production."""

    def __init__(self) -> None:
        self._records: dict[str, ScanRecord] = {}

    def get(self, scan_id: str) -> ScanRecord | None:
        return self._records.get(scan_id)

    def save(self, record: ScanRecord) -> None:
        self._records[record.id] = record


class ScanService:
    def __init__(
        self,
        settings: Settings,
        storage: StorageService,
        repository: ScanRepository,
        pipeline: VisionPipeline,
        scan_pipeline: ScanPipeline | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._repository = repository
        self._pipeline = pipeline
        self._scan_pipeline = scan_pipeline or ScanPipeline(
            ScannerConfig.from_settings(
                output_size=settings.board_output_size,
                margin_ratio=settings.classification_margin_ratio,
                model_path=settings.piece_classifier_path,
                stockfish_path=settings.stockfish_path,
                use_stockfish=settings.classification_use_stockfish_tiebreak,
            )
        )

    def get(self, scan_id: str) -> ScanRecord:
        record = self._repository.get(scan_id)
        if record is None:
            raise NotFoundError(f"Scan '{scan_id}' not found")
        return record

    async def process_upload(self, image: ValidatedImage) -> ScanRecord:
        scan_id = self._storage.new_scan_id()
        extension = self._storage.extension_for_mime(image.mime_type)
        original_path = self._storage.save_original(scan_id, image.data, extension)

        record = ScanRecord(
            id=scan_id,
            status=ScanStatus.PROCESSING,
            original_path=original_path,
            original_filename=image.filename,
            mime_type=image.mime_type,
        )
        self._repository.save(record)

        logger.info(
            "scan_started",
            scan_id=scan_id,
            filename=image.filename,
            bytes=image.size_bytes,
            mime=image.mime_type,
        )

        started = time.perf_counter()
        try:
            scan = self._scan_pipeline.run_from_bytes(image.data)
            detection_meta = {
                "corners": scan.corners_list,
                "homography": scan.homography_list(),
                "confidence": scan.confidence,
                "original_size": [scan.original_width, scan.original_height],
                "output_size": scan.output_width,
            }
            grid = scan.classification.dataset_grid
            classification = scan.classification
            if grid is None:
                raise RuntimeError("Scanner produced no grid")

            warped_bytes = cv2.imencode(".jpg", scan.warped_board)[1].tobytes()
            warped_path = self._storage.save_warped(scan_id, warped_bytes)

            record.status = ScanStatus.COMPLETED
            record.warped_path = warped_path
            record.fen = classification.interactive_fen or classification.fen
            record.fen_confidence = min(scan.confidence, classification.confidence)
            record.board_corners = [{"x": x, "y": y} for x, y in scan.corners_list]
            record.grid_metadata = {
                **(grid.to_metadata() if grid else {}),
                "classification": classification.to_metadata(),
                "scanner": scan.metadata,
            }
            record.pipeline_timing = scan.metadata.get("timing_ms", {})

            logger.info(
                "scan_completed",
                scan_id=scan_id,
                fen_confidence=record.fen_confidence,
                squares=len(grid.flat) if grid else 0,
            )
        except BoardNotFoundError as exc:
            record.status = ScanStatus.FAILED
            record.error_message = str(exc)
            logger.warning("scan_board_not_found", scan_id=scan_id, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            record.status = ScanStatus.FAILED
            record.error_message = f"Scan failed: {exc}"
            logger.exception("scan_failed", scan_id=scan_id)
        finally:
            record.processing_ms = int((time.perf_counter() - started) * 1000)
            self._repository.save(record)

        return record

