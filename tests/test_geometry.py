"""Bounding-box geometry helpers.

These are pure functions over ``(xmin, ymin, width, height)`` boxes and need
neither OpenCV nor a model, so they run anywhere numpy does.
"""
import numpy as np
import pytest

from TrackEverything import box_iou, cv2_bbox_reshape, cv2_bbox_to_tf_bbox


class TestBoxIou:
    def test_identical_boxes(self):
        assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert box_iou((0, 0, 10, 10), (50, 50, 10, 10)) == pytest.approx(0.0)

    def test_edge_touching_boxes_do_not_overlap(self):
        assert box_iou((0, 0, 10, 10), (10, 0, 10, 10)) == pytest.approx(0.0)

    def test_half_overlap_is_one_third(self):
        """Two 10x10 boxes sharing half their area: intersection 50, union 150."""
        assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)

    def test_full_containment(self):
        """A 1x1 box inside a 2x2 box: intersection 1, union 4."""
        assert box_iou((0, 0, 2, 2), (0, 0, 1, 1)) == pytest.approx(0.25)

    def test_is_symmetric(self):
        box_a, box_b = (0, 0, 10, 10), (3, 4, 8, 12)
        assert box_iou(box_a, box_b) == pytest.approx(box_iou(box_b, box_a))

    def test_zero_area_boxes_return_zero_not_zero_division(self):
        """Degenerate boxes used to raise ZeroDivisionError."""
        assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0

    def test_zero_area_against_real_box(self):
        assert box_iou((0, 0, 0, 0), (0, 0, 10, 10)) == pytest.approx(0.0)

    def test_result_is_a_plain_float(self):
        assert isinstance(box_iou((0, 0, 4, 4), (1, 1, 4, 4)), float)


class TestCv2BboxReshape:
    def test_known_value(self):
        assert cv2_bbox_reshape((10, 20, 30, 40)) == (10, 20, 40, 60)

    def test_zero_origin(self):
        assert cv2_bbox_reshape((0, 0, 5, 7)) == (0, 0, 5, 7)


class TestCv2BboxToTfBbox:
    def test_axis_order_is_swapped_to_y_first(self):
        """TF wants [y_min, x_min, y_max, x_max]."""
        result = cv2_bbox_to_tf_bbox((10, 20, 30, 40), width=100, height=200)
        assert result == pytest.approx([20 / 200, 10 / 100, 60 / 200, 40 / 100])

    def test_full_frame_box_normalises_to_unit_square(self):
        result = cv2_bbox_to_tf_bbox((0, 0, 100, 200), width=100, height=200)
        assert result == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_oversized_box_is_clamped(self):
        result = cv2_bbox_to_tf_bbox((0, 0, 500, 900), width=100, height=200)
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)

    def test_origin_box(self):
        result = cv2_bbox_to_tf_bbox((0, 0, 10, 10), width=100, height=100)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)

    def test_returns_ndarray(self):
        assert isinstance(cv2_bbox_to_tf_bbox((1, 2, 3, 4), 10, 10), np.ndarray)
