"""Unified production scanner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from vision.board.config import BoardDetectorConfig
from vision.board.grid_config import GridExtractorConfig
from vision.board.grid_homography import GridRectifierConfig
from vision.core.config import MlPipelineConfig
from vision.occupancy.config import OccupancyConfig
from vision.classification.pipeline import ClassificationPipelineConfig
from vision.classification.square_quality import DatasetSquareConfig
from vision.scanner.mode import ScannerMode


@dataclass(frozen=True, slots=True)
class LocalizationConfig:
    max_detection_dim: int = 1600
    blur_kernel: int = 5
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    canny_low: int = 40
    canny_high: int = 120
    max_mesh_retries: int = 2
    max_source_orthogonality_deg: float = 12.0
    max_source_spacing_cv: float = 0.35


@dataclass(frozen=True, slots=True)
class MeshRectifyConfig:
    output_size: int = 800
    max_aspect_deviation: float = 0.30
    min_cosine_angle: float = 0.25


@dataclass(frozen=True, slots=True)
class ScannerConfig:
    """Single config surface for the full grid-driven scanner."""

    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    mesh: MeshRectifyConfig = field(default_factory=MeshRectifyConfig)
    grid: GridExtractorConfig = field(default_factory=GridExtractorConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    ml: MlPipelineConfig = field(default_factory=MlPipelineConfig)
    dataset_square: DatasetSquareConfig = field(default_factory=DatasetSquareConfig)
    classification: ClassificationPipelineConfig = field(default_factory=ClassificationPipelineConfig)
    collect_debug: bool = True
    mode: ScannerMode = ScannerMode.FULL
    dataset_mode: bool = False
    dataset_output_dir: str | None = None
    use_stockfish_scoring: bool = True

    @classmethod
    def from_settings(
        cls,
        *,
        output_size: int = 800,
        margin_ratio: float = 0.10,
        upscale_size: int = 2048,
        model_path: str | None = None,
        stockfish_path: str | None = None,
        use_stockfish: bool = True,
        dataset_mode: bool = False,
    ) -> ScannerConfig:
        return cls(
            mesh=MeshRectifyConfig(output_size=output_size),
            grid=GridExtractorConfig(margin_ratio=margin_ratio, upscale_size=upscale_size),
            classification=ClassificationPipelineConfig(
                margin_ratio=margin_ratio,
                model_path=model_path,
                stockfish_path=stockfish_path,
                use_stockfish_tiebreak=use_stockfish,
            ),
            dataset_mode=dataset_mode,
            use_stockfish_scoring=use_stockfish,
        )

    def grid_rectifier(self) -> GridRectifierConfig:
        return GridRectifierConfig(
            output_size=self.mesh.output_size,
            max_aspect_deviation=self.mesh.max_aspect_deviation,
            min_cosine_angle=self.mesh.min_cosine_angle,
        )

    @classmethod
    def production(cls, **kwargs) -> ScannerConfig:
        """ML-first production config — requires trained ONNX models."""
        from dataclasses import replace as dc_replace

        base = cls.from_settings(**kwargs)
        return dc_replace(
            base,
            ml=MlPipelineConfig(
                require_occupancy_model=True,
                require_piece_model=True,
                allow_heuristic_fallback=False,
            ),
            occupancy=dc_replace(base.occupancy, ml_only=True),
        )

    @classmethod
    def piece_detection_only(cls, **kwargs) -> ScannerConfig:
        """YOLO localization + fine piece classifier on bbox crops."""
        from dataclasses import replace as dc_replace

        base = cls.from_settings(**kwargs)
        return dc_replace(
            base,
            mode=ScannerMode.PIECE_DETECTION_ONLY,
            ml=MlPipelineConfig(
                require_occupancy_model=False,
                require_piece_model=True,
                allow_heuristic_fallback=False,
                capture_ml_debug=False,
                use_stockfish_scoring=False,
            ),
            collect_debug=True,
        )

    def legacy_board_detector(self) -> BoardDetectorConfig:
        return BoardDetectorConfig(
            output_size=self.mesh.output_size,
            max_detection_dim=self.localization.max_detection_dim,
        )
