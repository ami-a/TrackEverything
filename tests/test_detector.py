"""End-to-end Detector behaviour, driven entirely by fakes."""
import numpy as np
import pytest
from conftest import FakeClassModel, FakeDetectionModel, tracker_factory

from TrackEverything import (
    ClassificationVars,
    DetectionVars,
    Detector,
    InspectorVars,
    StatisticalCalculator,
    StatMethods,
)

# [conf, x1, y1, x2, y2]
TWO_BOXES = [[0.9, 40, 40, 140, 200], [0.8, 200, 50, 280, 210]]


def build(detections=None, class_model=None, **ins_kwargs):
    ins_kwargs.setdefault("trck_type", tracker_factory())
    return Detector(
        det_vars=DetectionVars(
            detection_model=FakeDetectionModel(
                TWO_BOXES if detections is None else detections
            )
        ),
        class_vars=ClassificationVars(class_model=class_model),
        inspector_vars=InspectorVars(**ins_kwargs),
    )


class TestDetectionOnly:
    """No classification model: every object is class 0 with score 1.

    This path used to raise TypeError because the score matrix was 1-D, making
    each detection's class_score a bare float.
    """

    def test_summary_counts_both_objects(self, frame):
        detector = build()
        detector.update(frame)
        assert detector.get_current_class_summary() == {0: 2}

    def test_detections_have_ids(self, frame):
        detector = build()
        detector.update(frame)
        assert sorted(d.id_num for d in detector.detections) == [1, 2]

    def test_class_score_is_indexable(self, frame):
        """The smoothed score is the class score weighted by detection confidence."""
        detector = build(detections=[[0.9, 40, 40, 140, 200]])
        detector.update(frame)
        class_num, score = detector.detections[0].get_current_class()
        assert class_num == 0
        assert score == pytest.approx(0.9)


class TestWithClassification:
    def test_summary_uses_argmax(self, frame):
        detector = build(class_model=FakeClassModel(scores=(0.1, 0.9)))
        detector.update(frame)
        assert detector.get_current_class_summary() == {1: 2}

    def test_three_class_model_is_supported(self, frame):
        detector = build(class_model=FakeClassModel(scores=(0.1, 0.1, 0.8)))
        detector.update(frame)
        assert detector.get_current_class_summary() == {2: 2}

    def test_smoothing_accumulates_over_frames(self, frame):
        """EMA warms up from zero, so a steady signal still moves frame to frame."""
        detector = build(
            class_model=FakeClassModel(scores=(0.2, 0.8)),
            trck_resizing=False,
            stat_calc=StatisticalCalculator(method=StatMethods.EMA, class_num=2),
        )
        detector.update(frame)
        first = detector.detections[0].class_score.copy()
        for tracker in detector.trackers:
            tracker.tracker.boxes = [tracker.bounding_box]
        detector.update(frame)
        second = detector.detections[0].class_score
        assert not np.allclose(first, second)
        assert second[1] > first[1], "EMA should climb towards the steady signal"


class TestEdgeCases:
    def test_no_detections_does_not_crash(self, frame):
        detector = build(detections=[])
        detector.update(frame)
        assert detector.detections == []
        assert detector.get_current_class_summary() == {}

    def test_detections_below_threshold_are_filtered(self, frame):
        detector = build(detections=[[0.1, 40, 40, 140, 200]])
        detector.update(frame)
        assert detector.detections == []

    def test_ids_persist_across_frames(self, frame):
        detector = build(trck_resizing=False)
        detector.update(frame)
        first = sorted(d.id_num for d in detector.detections)
        for tracker in detector.trackers:
            tracker.tracker.boxes = [tracker.bounding_box]
        detector.update(frame)
        assert sorted(d.id_num for d in detector.detections) == first

    def test_two_detectors_do_not_share_ids(self, frame):
        one, two = build(), build()
        one.update(frame)
        two.update(frame)
        assert sorted(d.id_num for d in one.detections) == [1, 2]
        assert sorted(d.id_num for d in two.detections) == [1, 2]

    def test_missing_detection_model_raises(self):
        with pytest.raises(ValueError, match="detection_model"):
            Detector(det_vars=DetectionVars())

    def test_missing_classification_model_only_warns(self, capsys, frame):
        build()
        assert "Attention" in capsys.readouterr().out


class TestBoundingBoxMath:
    """get_detection_array used to compute width as xmax-confidence."""

    def test_width_and_height_come_from_the_right_indices(self, frame):
        detector = build(detections=[[0.9, 40, 50, 140, 210]])
        detector.update(frame)
        _xmin, _ymin, width, height = detector.detections[0].bounding_box
        assert width == pytest.approx(100)
        assert height == pytest.approx(160)

    def test_box_is_clamped_to_the_frame(self, frame):
        detector = build(detections=[[0.9, 0, 0, 10_000, 10_000]])
        detector.update(frame)
        _xmin, _ymin, width, height = detector.detections[0].bounding_box
        assert width <= frame.shape[1]
        assert height <= frame.shape[0]
