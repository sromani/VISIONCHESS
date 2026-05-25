"""Board detection errors."""


class BoardDetectionError(Exception):
    """Base error for the board detection pipeline."""


class BoardNotFoundError(BoardDetectionError):
    """No valid chess board quadrilateral was found in the image."""


class InvalidCornersError(BoardDetectionError):
    """Detected corners do not form a usable quadrilateral."""


class InvalidGridError(BoardDetectionError):
    """Warped board cannot be split into a valid 8×8 grid."""
