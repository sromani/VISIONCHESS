#!/usr/bin/env python3
"""Download chesscog transfer-learning real OTB photos (~27 test images)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = BENCHMARK_ROOT / "_tmp_tl.zip"
OUT_DIR = BENCHMARK_ROOT / "images" / "chesscog_transfer_learning"
GDRIVE_ID = "1Z9fTXRb7FlqzgTTXoywgiQP-Z1cH1v3W"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _has_test_images():
        _download_zip()
        _extract_zip()
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()

    _cleanup_macos_artifacts(OUT_DIR)
    print(f"Dataset ready at {OUT_DIR / 'test'}")
    print("Next: python benchmark/scripts/import_chesscog_transfer_learning.py")


def _has_test_images() -> bool:
    test_dir = OUT_DIR / "test"
    return test_dir.exists() and any(test_dir.glob("*.png"))


def _download_zip() -> None:
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    print(f"Downloading chesscog transfer-learning dataset ({GDRIVE_ID})...")
    gdown.download(id=GDRIVE_ID, output=str(ZIP_PATH), quiet=False)


def _extract_zip() -> None:
    print(f"Extracting to {OUT_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(OUT_DIR)


def _cleanup_macos_artifacts(root: Path) -> None:
    macos = root / "__MACOSX"
    if macos.exists():
        shutil.rmtree(macos)


if __name__ == "__main__":
    main()
