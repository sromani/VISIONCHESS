"""Human-readable benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path

from vision.benchmark.metrics import ALL_LABELS, BenchmarkAggregate


def print_summary(aggregate: BenchmarkAggregate) -> None:
    print("=" * 60)
    print("REAL-PHOTO BENCHMARK")
    print("=" * 60)
    print(f"Cases run:           {aggregate.cases_run}")
    print(f"Cases failed:        {aggregate.cases_failed}")
    print(f"Piece accuracy:      {aggregate.piece_accuracy:.2%}")
    print(f"Per-square accuracy: {aggregate.per_square_accuracy:.2%}")
    print(f"Occupancy F1:        {aggregate.occupancy_f1:.2%}")
    print(f"Occupancy precision: {aggregate.occupancy_precision:.2%}")
    print(f"Occupancy recall:    {aggregate.occupancy_recall:.2%}")
    print(f"Full-board accuracy: {aggregate.full_board_accuracy:.2%}")
    print(f"Legal FEN rate:      {aggregate.legal_fen_rate:.2%}")
    print("=" * 60)
    _print_confusion_matrix(aggregate)
    print()
    _print_worst_cases(aggregate, n=5)


def write_json_report(aggregate: BenchmarkAggregate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregate.to_dict(), indent=2), encoding="utf-8")


def _print_confusion_matrix(aggregate: BenchmarkAggregate) -> None:
    print("\nConfusion matrix (rows=expected, cols=predicted):")
    short = [_short_label(l) for l in ALL_LABELS]
    header = "           " + " ".join(f"{s:>4}" for s in short)
    print(header)
    for exp in ALL_LABELS:
        row = aggregate.confusion[exp]
        cells = " ".join(f"{row[p]:4d}" for p in ALL_LABELS)
        print(f"{_short_label(exp):>10} {cells}")


def _short_label(label: str) -> str:
    if label == "empty":
        return " emp"
    parts = label.split("_", 1)
    color = "W" if parts[0] == "white" else "B"
    piece_map = {
        "pawn": "P",
        "knight": "N",
        "bishop": "B",
        "rook": "R",
        "queen": "Q",
        "king": "K",
    }
    return f"{color}{piece_map.get(parts[1], '?')}"


def _print_worst_cases(aggregate: BenchmarkAggregate, *, n: int) -> None:
    ok = [c for c in aggregate.case_results if c.pipeline_error is None]
    if not ok:
        return
    ranked = sorted(ok, key=lambda c: c.per_square_accuracy)
    print(f"Worst {min(n, len(ranked))} cases by per-square accuracy:")
    for case in ranked[:n]:
        errs = sum(1 for s in case.per_square if not s.label_match)
        print(
            f"  {case.case_id}: {case.per_square_accuracy:.1%} "
            f"({errs}/64 wrong) — {Path(case.image_path).name}"
        )
