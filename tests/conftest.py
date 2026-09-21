"""Shared fakes and fixtures.

Everything here exists so the suite can exercise the real matching, statistics
and drawing code without TensorFlow, a GPU, a video file or a network.
"""
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pytest


class FakeTracker:
    """Implements the two-method OpenCV tracker contract: ``init`` and ``update``.

    Args:
        boxes: successive boxes to return from ``update``. The last one repeats.
        init_ok: what ``init`` reports. ``None`` mimics OpenCV >= 4.5.1, which
            returns ``None`` rather than a bool.
        update_ok: what ``update`` reports.
    """

    def __init__(self, boxes: Optional[Sequence] = None, init_ok=True, update_ok=True):
        self.boxes = list(boxes or [])
        self.init_ok = init_ok
        self.update_ok = update_ok
        self.bbox = None
        self.init_calls = 0
        self.update_calls = 0

    def init(self, frame, bbox):
        self.init_calls += 1
        self.bbox = tuple(bbox)
        return self.init_ok

    def update(self, frame):
        self.update_calls += 1
        if self.boxes:
            self.bbox = self.boxes.pop(0)
        return self.update_ok, self.bbox


def tracker_factory(**kwargs):
    """Return a zero-arg callable suitable for ``InspectorVars(trck_type=...)``."""
    return lambda: FakeTracker(**kwargs)


class FakeDetectionModel:
    """``.predict(batch) -> [[conf, x1, y1, x2, y2], ...]``.

    Args:
        detections: rows to return, or a list of per-frame row-lists to return in turn.
    """

    def __init__(self, detections: Optional[list[list]] = None, per_frame=None):
        self.detections = detections if detections is not None else []
        self.per_frame = per_frame
        self.calls = 0

    def predict(self, batch):
        result = self.detections
        if self.per_frame is not None:
            idx = min(self.calls, len(self.per_frame) - 1)
            result = self.per_frame[idx]
        self.calls += 1
        return result


class FakeClassModel:
    """``.predict(images) -> np.ndarray`` of shape ``(n, k)``."""

    def __init__(self, scores=(0.2, 0.8)):
        self.scores = np.asarray(scores, dtype=float)
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        return np.tile(self.scores, (len(images), 1))


@pytest.fixture
def frame():
    """A small BGR frame with two bright rectangles a tracker can lock onto."""
    img = np.zeros((240, 360, 3), dtype=np.uint8)
    img[40:200, 40:140] = 200
    img[50:210, 200:280] = 120
    return img
