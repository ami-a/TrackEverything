"""Non-max suppression over detections."""
import numpy as np
import pytest

from TrackEverything import DetectedObj, non_max_suppressions


def det(score, bbox):
    """Build a DetectedObj with only the fields NMS reads."""
    return DetectedObj(score, np.array([1.0]), bbox)


def test_empty_list_passes_through():
    assert non_max_suppressions([]) == []


def test_single_detection_survives():
    result = non_max_suppressions([det(0.9, (0, 0, 10, 10))])
    assert len(result) == 1


def test_higher_score_wins_on_identical_boxes():
    low, high = det(0.4, (0, 0, 10, 10)), det(0.9, (0, 0, 10, 10))
    result = non_max_suppressions([low, high])
    assert len(result) == 1
    assert result[0] is high


def test_order_does_not_change_the_winner():
    low, high = det(0.4, (0, 0, 10, 10)), det(0.9, (0, 0, 10, 10))
    assert non_max_suppressions([high, low])[0] is high


def test_boxes_below_threshold_both_survive():
    """IOU of 1/3 with a threshold of 0.5 keeps both."""
    result = non_max_suppressions(
        [det(0.9, (0, 0, 10, 10)), det(0.8, (5, 0, 10, 10))], threshold_iou=0.5
    )
    assert len(result) == 2


def test_distant_boxes_both_survive():
    result = non_max_suppressions(
        [det(0.9, (0, 0, 10, 10)), det(0.8, (100, 100, 10, 10))]
    )
    assert len(result) == 2


def test_vector_scores_use_max():
    """The Iterable branch: a vector det_score is compared by its maximum."""
    weak = det(np.array([0.1, 0.2]), (0, 0, 10, 10))
    strong = det(np.array([0.05, 0.95]), (0, 0, 10, 10))
    result = non_max_suppressions([weak, strong])
    assert len(result) == 1
    assert result[0] is strong


def test_mutates_the_caller_list_in_place():
    """Documented behaviour that assign_detections_to_trackers relies on."""
    detections = [det(0.9, (0, 0, 10, 10)), det(0.4, (0, 0, 10, 10))]
    returned = non_max_suppressions(detections)
    assert returned is detections
    assert len(detections) == 1


def test_three_overlapping_boxes_keep_only_the_best():
    detections = [
        det(0.5, (0, 0, 10, 10)),
        det(0.95, (1, 1, 10, 10)),
        det(0.6, (2, 2, 10, 10)),
    ]
    result = non_max_suppressions(detections, threshold_iou=0.3)
    assert len(result) == 1
    assert result[0].det_score == pytest.approx(0.95)
