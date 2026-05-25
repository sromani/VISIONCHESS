"""Unit tests for benchmark utilities (no pipeline)."""

from __future__ import annotations

from vision.benchmark.fen_grid import flip_grid_vertical, labels_to_placement, placement_to_labels
from vision.benchmark.metrics import compare_grids


def test_placement_roundtrip_starting_position() -> None:
    placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    grid = placement_to_labels(placement)
    assert labels_to_placement(grid) == placement


def test_compare_grids_perfect_match() -> None:
    placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    grid = placement_to_labels(placement)
    comparisons, metrics = compare_grids(grid, grid)
    assert len(comparisons) == 64
    assert metrics["per_square_accuracy"] == 1.0
    assert metrics["piece_accuracy"] == 1.0
    assert metrics["occupancy_f1"] == 1.0


def test_compare_grids_one_wrong_piece() -> None:
    expected = placement_to_labels("8/8/8/8/8/8/8/4K3")
    predicted = placement_to_labels("8/8/8/8/8/8/8/4Q3")
    _, metrics = compare_grids(expected, predicted)
    assert metrics["piece_accuracy"] == 0.0
    assert metrics["per_square_accuracy"] == 63 / 64


def test_flip_grid_vertical_swaps_ranks() -> None:
    grid = placement_to_labels("8/8/8/4P3/8/8/8/8")
    flipped = flip_grid_vertical(grid)
    assert flipped[0] == grid[7]
    assert flipped[7] == grid[0]
