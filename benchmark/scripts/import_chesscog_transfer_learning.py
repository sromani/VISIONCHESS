#!/usr/bin/env python3
"""Build manifest.jsonl from chesscog transfer-learning real photos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = BENCHMARK_ROOT / "images" / "chesscog_transfer_learning"
TEST_DIR = IMAGES_ROOT / "test"
MANIFEST = BENCHMARK_ROOT / "manifest.jsonl"


def main() -> None:
    if not TEST_DIR.exists():
        print(f"Missing dataset at {TEST_DIR}")
        print("Run: python benchmark/scripts/download_real_photos.py")
        sys.exit(1)

    lines: list[str] = []
    for png in sorted(TEST_DIR.glob("*.png")):
        meta_path = png.with_suffix(".json")
        if not meta_path.exists():
            print(f"Skip {png.name}: no .json sidecar")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        placement = meta["fen"].split()[0]
        rel = png.relative_to(BENCHMARK_ROOT).as_posix()
        entry = {
            "id": f"chesscog_tl_{png.stem}",
            "image": rel,
            "placement": placement,
            "source": "chesscog_transfer_learning",
            "tags": ["real_photo", "otb", "chesscog"],
            "active_color": "w" if meta.get("white_turn") else "b",
            "notes": "Physical chess set — chesscog transfer learning benchmark (27 photos)",
        }
        lines.append(json.dumps(entry, ensure_ascii=False))

    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} cases -> {MANIFEST}")


if __name__ == "__main__":
    main()
