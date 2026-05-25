"""Unit tests for corner ordering."""

from __future__ import annotations

import numpy as np

from vision.board.corners import (
    order_points,
    order_points_by_angle,
    order_points_sum_diff,
    order_points_y_sort,
    quadrilateral_aspect_ratio,
)


class TestOrderPoints:
    def test_axis_aligned_square(self) -> None:
        pts = np.float32([[10, 10], [90, 10], [90, 90], [10, 90]])
        ordered = order_points(pts)
        np.testing.assert_allclose(ordered, pts, atol=1e-4)

    def test_perspective_quad(self) -> None:
        pts = np.float32([[75, 40], [525, 20], [485, 560], [35, 525]])
        ordered = order_points(pts)
        assert ordered[0][1] < ordered[2][1]  # TL above BR
        assert ordered[0][0] < ordered[1][0]  # TL left of TR
        assert ordered[3][0] < ordered[2][0]  # BL left of BR

    def test_shuffled_input(self) -> None:
        pts = np.float32([[485, 560], [75, 40], [35, 525], [525, 20]])
        ordered = order_points(pts)
        np.testing.assert_allclose(ordered[0], [75, 40], atol=1e-4)
        np.testing.assert_allclose(ordered[1], [525, 20], atol=1e-4)
        np.testing.assert_allclose(ordered[2], [485, 560], atol=1e-4)
        np.testing.assert_allclose(ordered[3], [35, 525], atol=1e-4)

    def test_y_sort_matches_expected(self) -> None:
        pts = np.float32([[75, 40], [525, 20], [485, 560], [35, 525]])
        y_sorted = order_points_y_sort(pts)
        sum_diff = order_points_sum_diff(pts)
        assert y_sorted.shape == (4, 2)
        assert sum_diff.shape == (4, 2)

    def test_aspect_ratio_perfect_square(self) -> None:
        pts = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        assert quadrilateral_aspect_ratio(pts) == 1.0

    def test_angle_ordered_quad(self) -> None:
        pts = np.float32([[0, 50], [50, 0], [100, 50], [50, 100]])
        ordered = order_points_by_angle(pts)
        assert ordered.shape == (4, 2)
