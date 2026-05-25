"""Board classification: split → classify → orient → FEN."""

from vision.classification.pipeline import ClassificationPipeline, ClassificationPipelineConfig
from vision.classification.types import ClassificationResult, SquareClassification

__all__ = [
    "ClassificationPipeline",
    "ClassificationPipelineConfig",
    "ClassificationResult",
    "SquareClassification",
]
