"""Board validation — python-chess legality + Stockfish plausibility."""

from vision.validation.scorer import BoardValidator, score_hypothesis_with_stockfish

__all__ = ["BoardValidator", "score_hypothesis_with_stockfish"]
