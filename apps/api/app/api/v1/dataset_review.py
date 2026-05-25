"""Human-in-the-loop dataset review API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_storage_service
from app.utils.file_storage import StorageService
from vision.classification.labels import CLASS_NAMES
from vision.dataset.builder import DatasetBuilder

router = APIRouter(prefix="/dataset", tags=["dataset"])


class RelabelRequest(BaseModel):
    sample_path: str = Field(..., description="Relative path under dataset root")
    new_label: str


@router.get("/{job_id}/samples")
async def list_samples(
    job_id: str,
    storage: StorageService = Depends(get_storage_service),
) -> dict:
    root = storage.job_directory(job_id) / "dataset" / job_id / "train"
    if not root.exists():
        raise HTTPException(status_code=404, detail="Dataset not found for job")

    samples: list[dict] = []
    for label in CLASS_NAMES:
        class_dir = root / label
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.png"))[:32]:
            samples.append({"label": label, "path": str(path.relative_to(root.parent.parent))})
    return {"job_id": job_id, "samples": samples}


@router.post("/{job_id}/relabel")
async def relabel_sample(
    job_id: str,
    body: RelabelRequest,
    storage: StorageService = Depends(get_storage_service),
) -> dict:
    if body.new_label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASS_NAMES}")

    dataset_root = storage.job_directory(job_id) / "dataset" / job_id
    sample = storage.job_directory(job_id) / body.sample_path
    if not sample.exists():
        raise HTTPException(status_code=404, detail="Sample not found")

    builder = DatasetBuilder(dataset_root, split="train")
    dest = builder.relabel(sample, body.new_label)
    return {"ok": True, "new_path": str(dest)}
