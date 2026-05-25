"""Save square crops into class-folder dataset layout."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.classification.labels import CLASS_NAMES, FEN_SYMBOL_TO_LABEL


@dataclass(frozen=True, slots=True)
class DatasetSample:
    path: Path
    label: str
    square_name: str
    confidence: float
    orientation: str
    job_id: str
    source: str


class DatasetBuilder:
    """Build ``dataset/<split>/<class>/*.png`` from scans or auto-labeling."""

    def __init__(self, root: Path, *, split: str = "train") -> None:
        self.root = root
        self.split = split
        self._ensure_class_dirs()

    def _ensure_class_dirs(self) -> None:
        for name in CLASS_NAMES:
            (self.root / self.split / name).mkdir(parents=True, exist_ok=True)

    def save_square(
        self,
        image: NDArray[np.uint8],
        label: str,
        *,
        square_name: str,
        confidence: float,
        orientation: str = "normal",
        job_id: str = "",
        source: str = "scan",
    ) -> DatasetSample:
        if label not in CLASS_NAMES:
            label = "empty"
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{job_id}_{square_name}_{ts}.png" if job_id else f"{square_name}_{ts}.png"
        path = self.root / self.split / label / filename
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(path), bgr)
        sample = DatasetSample(
            path=path,
            label=label,
            square_name=square_name,
            confidence=confidence,
            orientation=orientation,
            job_id=job_id,
            source=source,
        )
        self._append_manifest(sample)
        return sample

    def save_from_scan(
        self,
        dataset_grid,
        squares,
        *,
        job_id: str,
        orientation: str = "normal",
        analysis_grid=None,
        occupancy: dict | None = None,
    ) -> list[DatasetSample]:
        saved: list[DatasetSample] = []
        by_name = {sq.square_name: sq for sq in squares}
        for crop in dataset_grid.flat:
            cls = by_name.get(crop.square_name)
            if cls is None:
                continue
            label = cls.label if cls.occupied else "empty"
            conf = cls.confidence if cls.occupied else cls.occupancy_score
            occ_prob = occupancy.get(crop.square_name).probability if occupancy and crop.square_name in occupancy else None
            saved.append(
                self.save_square(
                    crop.image,
                    label,
                    square_name=crop.square_name,
                    confidence=conf,
                    orientation=orientation,
                    job_id=job_id,
                    source="scan",
                )
            )
            if analysis_grid is not None:
                analysis_crop = next((s for s in analysis_grid.flat if s.square_name == crop.square_name), None)
                if analysis_crop is not None:
                    hi_dir = self.root / self.split / label / "analysis"
                    hi_dir.mkdir(parents=True, exist_ok=True)
                    ts = saved[-1].path.stem
                    hi_path = hi_dir / f"{ts}_hires.png"
                    bgr = analysis_crop.image if analysis_crop.image.ndim == 3 else analysis_crop.image
                    cv2.imwrite(str(hi_path), bgr)
            if occ_prob is not None:
                self._append_occupancy_record(crop.square_name, occ_prob, label, job_id)
        return saved

    def _append_occupancy_record(self, square_name: str, occ_prob: float, label: str, job_id: str) -> None:
        log = self.root / "occupancy.jsonl"
        record = {"job_id": job_id, "square": square_name, "occupancy_prob": occ_prob, "label": label}
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def relabel(self, sample_path: Path, new_label: str) -> Path:
        """Move sample to corrected class folder (human review)."""
        if new_label not in CLASS_NAMES:
            msg = f"Invalid label: {new_label}"
            raise ValueError(msg)
        sample_path = Path(sample_path)
        dest = self.root / self.split / new_label / sample_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(sample_path), str(dest))
        self._log_correction(sample_path, dest, new_label)
        return dest

    def _append_manifest(self, sample: DatasetSample) -> None:
        manifest = self.root / "manifest.jsonl"
        record = asdict(sample)
        record["path"] = str(sample.path.relative_to(self.root))
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _log_correction(self, old: Path, new: Path, label: str) -> None:
        log = self.root / "corrections.jsonl"
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "from": str(old),
            "to": str(new),
            "label": label,
        }
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def label_from_fen_square(symbol: str | None) -> str:
    if not symbol:
        return "empty"
    return FEN_SYMBOL_TO_LABEL.get(symbol, "empty")
