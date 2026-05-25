"""Persist scan squares into class-folder dataset."""

from __future__ import annotations

from pathlib import Path

from vision.dataset.builder import DatasetBuilder
from vision.scanner.context import ScanContext


def record_scan(ctx: ScanContext, job_id: str, output_dir: Path) -> Path:
    root = output_dir / "dataset" / job_id
    builder = DatasetBuilder(root, split="train")
    orientation = ctx.metadata.get("validation", {}).get("orientation", "normal")

    if ctx.dataset_grid is not None and ctx.squares:
        builder.save_from_scan(
            ctx.dataset_grid,
            ctx.squares,
            job_id=job_id,
            orientation=orientation,
            analysis_grid=ctx.analysis_grid,
            occupancy=ctx.occupancy,
        )

    return root
