"""Map YOLO bounding-box centers to chess squares."""

from __future__ import annotations

from vision.board.types import BoardGridResult
from vision.inference.yolo_detector import YoloPieceDetection


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def center_to_square_name(cx: float, cy: float, board_size: int) -> str:
    """Map pixel center to algebraic square (row 0 = rank 8)."""
    cell = board_size / 8.0
    col = min(7, max(0, int(cx / cell)))
    row = min(7, max(0, int(cy / cell)))
    rank = 8 - row
    file_char = chr(ord("a") + col)
    return f"{file_char}{rank}"


def assign_squares(
    detections: list[YoloPieceDetection],
    board_size: int,
) -> list[YoloPieceDetection]:
    """Attach square_name from bbox center."""
    assigned: list[YoloPieceDetection] = []
    for det in detections:
        cx, cy = bbox_center(det.bbox)
        square = center_to_square_name(cx, cy, board_size)
        assigned.append(
            YoloPieceDetection(
                label=det.label,
                confidence=det.confidence,
                bbox=det.bbox,
                square_name=square,
            )
        )
    return assigned


def resolve_square_assignments(
    detections: list[YoloPieceDetection],
    grid: BoardGridResult,
) -> dict[str, YoloPieceDetection]:
    """One piece per square — highest confidence wins."""
    board_size = grid.board_size
    with_squares = assign_squares(detections, board_size)
    best: dict[str, YoloPieceDetection] = {}
    for det in with_squares:
        if det.square_name is None:
            continue
        prev = best.get(det.square_name)
        if prev is None or det.confidence > prev.confidence:
            best[det.square_name] = det
    return best


def build_classified_square_records(
    grid: BoardGridResult,
    classified_by_square: dict[str, dict],
) -> list[dict]:
    """64 square records using classifier labels (not YOLO fine class)."""
    records: list[dict] = []
    for sq in grid.flat:
        det = classified_by_square.get(sq.square_name)
        occupied = det is not None
        label = det.get("classified_label", "empty") if det else "empty"
        conf = float(det.get("classified_confidence", 0.0)) if det else 0.0
        loc_conf = float(det.get("localization_confidence", 0.0)) if det else 0.0
        bbox = det.get("bbox") if det else None
        records.append(
            {
                "square_name": sq.square_name,
                "row": sq.row,
                "col": sq.col,
                "occupied": occupied,
                "label": label if occupied else "empty",
                "confidence": conf,
                "piece_label": label if occupied else "empty",
                "piece_confidence": conf,
                "localization_confidence": loc_conf,
                "bbox": bbox,
                "top3": det.get("top3", []) if det else [],
                "cell_bbox": list(sq.cell_bbox),
            }
        )
    return records


def build_square_records(
    grid: BoardGridResult,
    assigned: dict[str, YoloPieceDetection],
) -> list[dict]:
    """64 square records for API + synthetic board render."""
    records: list[dict] = []
    for sq in grid.flat:
        det = assigned.get(sq.square_name)
        occupied = det is not None
        label = det.label if det else "empty"
        conf = det.confidence if det else 0.0
        bbox = list(det.bbox) if det else None
        records.append(
            {
                "square_name": sq.square_name,
                "row": sq.row,
                "col": sq.col,
                "occupied": occupied,
                "label": label,
                "confidence": conf,
                "piece_label": label,
                "piece_confidence": conf,
                "bbox": bbox,
                "cell_bbox": list(sq.cell_bbox),
            }
        )
    return records
