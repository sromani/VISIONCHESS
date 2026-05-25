"""Load benchmark manifest.jsonl entries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    image: Path
    placement: str
    source: str = "unknown"
    tags: tuple[str, ...] = ()
    active_color: str | None = None
    notes: str | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_manifest_path() -> Path:
    return repo_root() / "benchmark" / "manifest.jsonl"


def default_benchmark_root() -> Path:
    return repo_root() / "benchmark"


def load_manifest(path: Path | None = None) -> list[BenchmarkCase]:
    manifest_path = path or default_manifest_path()
    if not manifest_path.exists():
        msg = f"Benchmark manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    root = manifest_path.parent
    cases: list[BenchmarkCase] = []
    for line_no, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        data = json.loads(line)
        image_rel = data["image"]
        image_path = Path(image_rel)
        if not image_path.is_absolute():
            image_path = root / image_rel
        if "placement" not in data and "fen" in data:
            placement = str(data["fen"]).split()[0]
        else:
            placement = str(data["placement"])

        cases.append(
            BenchmarkCase(
                id=str(data.get("id", image_path.stem)),
                image=image_path,
                placement=placement,
                source=str(data.get("source", "unknown")),
                tags=tuple(data.get("tags", [])),
                active_color=data.get("active_color"),
                notes=data.get("notes"),
            )
        )
    return cases
