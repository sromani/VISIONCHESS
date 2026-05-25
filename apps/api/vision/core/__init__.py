"""Core types and protocols for the modular chess vision pipeline."""

from vision.core.config import MlPipelineConfig
from vision.core.types import BoardHypothesis, SoftSquarePrediction

__all__ = ["BoardHypothesis", "MlPipelineConfig", "SoftSquarePrediction"]
