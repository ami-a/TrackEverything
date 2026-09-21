"""Detection-to-tracker matching, identity persistence and tracker lifecycle."""
import numpy as np
import pytest
from conftest import tracker_factory

from TrackEverything import (
    DetectedObj,
    InspectorVars,
    TrackerObj,
    assign_detections_to_trackers,
    update_trackers,
)


def det(bbox, score=0.9, classes=(0.2, 0.8)):
    return DetectedObj(score, np.array(classes, dtype=float), bbox)


def inspector_vars(**kwargs):
    kwargs.setdefault("trck_type", tracker_factory())
    return InspectorVars(**kwargs)


class TestTrackerCreation:
    def test_first_detection_creates_one_tracker_with_id_one(self, frame):
        ins = inspector_vars()
        detections, trackers = assign_detections_to_trackers(
            [], [det((40, 40, 100, 160))], frame, ins
        )
        assert len(trackers) == 1
        assert trackers[0].id_num == 1
        assert detections[0].id_num == 1

    def test_each_detection_gets_a_distinct_id(self, frame):
        ins = inspector_vars()
        _, trackers = assign_detections_to_trackers(
            [], [det((40, 40, 100, 160)), det((200, 50, 80, 160))], frame, ins
        )
        assert sorted(t.id_num for t in trackers) == [1, 2]

    def test_init_returning_none_is_treated_as_success(self, frame):
        """OpenCV >= 4.5.1 returns None from init(); `not None` would fail it."""
        ins = inspector_vars(trck_type=tracker_factory(init_ok=None))
        _, trackers = assign_detections_to_trackers(
            [], [det((40, 40, 100, 160))], frame, ins
        )
        assert len(trackers) == 1
        assert trackers[0].fails == 0

    def test_failed_init_marks_tracker_destroyable(self, frame):
        ins = inspector_vars(trck_type=tracker_factory(init_ok=False))
        tracker = TrackerObj(1, frame, (40, 40, 100, 160), ins)
        assert tracker.destroy()

    def test_failed_init_tracker_is_never_kept(self, frame):
        ins = inspector_vars(trck_type=tracker_factory(init_ok=False))
        _, trackers = assign_detections_to_trackers(
            [], [det((40, 40, 100, 160))], frame, ins
        )
        assert trackers == []

    def test_bounding_box_is_coerced_to_ints_for_opencv(self, frame):
        """Models emit floats; OpenCV rejects them with an opaque overload error."""
        ins = inspector_vars()
        tracker = TrackerObj(1, frame, (40.7, 40.2, 100.9, 160.4), ins)
        assert all(isinstance(v, int) for v in tracker.tracker.bbox)


class TestIdentityPersistence:
    def test_same_object_keeps_its_id_across_frames(self, frame):
        """The core promise of the library."""
        ins = inspector_vars(trck_resizing=False)
        box = (40, 40, 100, 160)
        _, trackers = assign_detections_to_trackers([], [det(box)], frame, ins)
        first_id = trackers[0].id_num

        for tracker in trackers:
            tracker.tracker.boxes = [box]
        detections, trackers = assign_detections_to_trackers(
            trackers, [det(box)], frame, ins
        )
        assert len(trackers) == 1
        assert trackers[0].id_num == first_id
        assert detections[0].id_num == first_id

    def test_resizing_preserves_id_and_statistics(self, frame):
        ins = inspector_vars(trck_resizing=True)
        box = (40, 40, 100, 160)
        _, trackers = assign_detections_to_trackers([], [det(box)], frame, ins)
        original_id = trackers[0].id_num
        original_stats = trackers[0].statistical_calc

        for tracker in trackers:
            tracker.tracker.boxes = [box]
        _, trackers = assign_detections_to_trackers(trackers, [det(box)], frame, ins)
        assert trackers[0].id_num == original_id
        assert trackers[0].statistical_calc is original_stats


class TestMatchingQuality:
    def test_hungarian_beats_greedy_on_a_crossing_pair(self, frame):
        """Constructed so a greedy nearest-match would swap the two identities.

        Greedy would give tracker A its best box and leave B the leftover;
        the optimal assignment maximises total IOU instead.
        """
        ins = inspector_vars(trck_resizing=False, iou_paring_threshold=0.01)
        box_a, box_b = (0, 0, 100, 100), (60, 0, 100, 100)
        _, trackers = assign_detections_to_trackers(
            [], [det(box_a), det(box_b)], frame, ins
        )
        by_id = {t.id_num: t for t in trackers}
        assert len(by_id) == 2

        #each tracker reports back the box it was created on, so the optimal
        #assignment is the identity mapping rather than the crossed one
        for tracker in trackers:
            tracker.tracker.boxes = [tracker.bounding_box]
        detections, trackers = assign_detections_to_trackers(
            trackers, [det(box_a), det(box_b)], frame, ins
        )
        assert sorted(d.id_num for d in detections) == [1, 2]
        assert len(trackers) == 2, "optimal matching should not spawn extra trackers"

    def test_low_iou_creates_a_new_tracker_instead_of_matching(self, frame):
        ins = inspector_vars(trck_resizing=False, iou_paring_threshold=0.9)
        box = (40, 40, 100, 160)
        _, trackers = assign_detections_to_trackers([], [det(box)], frame, ins)

        far_box = (250, 150, 60, 60)
        for tracker in trackers:
            tracker.tracker.boxes = [box]
        _, trackers = assign_detections_to_trackers(trackers, [det(far_box)], frame, ins)
        assert len(trackers) >= 1
        assert any(t.id_num == 2 for t in trackers)

    def test_nms_is_skipped_for_negative_threshold(self, frame):
        ins = inspector_vars()
        box = (40, 40, 100, 160)
        detections, _ = assign_detections_to_trackers(
            [], [det(box), det(box)], frame, ins, iou_overlapping_threshold=-1
        )
        assert len(detections) == 2

    def test_nms_removes_duplicates_when_enabled(self, frame):
        ins = inspector_vars()
        box = (40, 40, 100, 160)
        detections, _ = assign_detections_to_trackers(
            [], [det(box, score=0.9), det(box, score=0.4)],
            frame, ins, iou_overlapping_threshold=0.3,
        )
        assert len(detections) == 1


class TestLifecycle:
    def test_missing_detection_accumulates_failures(self, frame):
        ins = inspector_vars(max_trck_fails=100)
        _, trackers = assign_detections_to_trackers(
            [], [det((40, 40, 100, 160))], frame, ins
        )
        before = trackers[0].fails
        detections, trackers = assign_detections_to_trackers(trackers, [], frame, ins)
        assert detections == []
        assert trackers[0].fails > before

    def test_tracker_is_dropped_after_too_many_failures(self, frame):
        ins = inspector_vars(max_trck_fails=2)
        _, trackers = assign_detections_to_trackers(
            [], [det((40, 40, 100, 160))], frame, ins
        )
        for _ in range(5):
            _, trackers = assign_detections_to_trackers(trackers, [], frame, ins)
        assert trackers == []

    def test_update_trackers_rewards_success(self, frame):
        ins = inspector_vars()
        tracker = TrackerObj(1, frame, (40, 40, 100, 160), ins)
        tracker.fails = 5
        update_trackers(frame, [tracker])
        assert tracker.fails == pytest.approx(5 - ins.trck_reward_pt)

    def test_update_trackers_penalises_failure(self, frame):
        ins = inspector_vars(trck_type=tracker_factory(update_ok=False))
        tracker = TrackerObj(1, frame, (40, 40, 100, 160), ins)
        update_trackers(frame, [tracker])
        assert tracker.fails == pytest.approx(ins.trck_failure_pt)

    def test_update_trackers_filters_destroyed(self, frame):
        ins = inspector_vars(max_trck_fails=1, trck_type=tracker_factory(update_ok=False))
        tracker = TrackerObj(1, frame, (40, 40, 100, 160), ins)
        result = update_trackers(frame, [tracker], penalties=10)
        assert result == []
