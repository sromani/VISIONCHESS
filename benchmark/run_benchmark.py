#!/usr/bin/env python3
"""Run real-photo benchmark and print metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from vision.benchmark.manifest import default_manifest_path  # noqa: E402
from vision.benchmark.report import print_summary, write_json_report  # noqa: E402
from vision.benchmark.runner import RealPhotoBenchmarkRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionChess real-photo FEN benchmark")
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--limit", type=int, default=None, help="Max cases to run")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmark" / "results" / "latest.json",
    )
    args = parser.parse_args()

    runner = RealPhotoBenchmarkRunner(manifest_path=args.manifest)
    aggregate = runner.run_all(limit=args.limit)
    print_summary(aggregate)
    write_json_report(aggregate, args.output)
    print(f"\nJSON report: {args.output}")
    failed = aggregate.cases_failed
    if failed == aggregate.cases_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
