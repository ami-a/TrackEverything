"""Tracker lookup across the OpenCV 4.5.1 namespace split.

BOOSTING, TLD, MedianFlow and MOSSE moved from ``cv2`` to ``cv2.legacy`` in
OpenCV 4.5.1. Resolution must find them either way, be case-insensitive, and
fail with a message that says what to do.
"""
from types import SimpleNamespace

import pytest

from TrackEverything import InspectorVars, available_trackers, get_tracker, tool_box


def test_csrt_resolves():
    assert callable(get_tracker("CSRT"))


def test_lookup_is_case_insensitive():
    """get_tracker('csrt') used to return None and fail much later."""
    assert get_tracker("csrt") is get_tracker("CSRT") is get_tracker("CsRt")


def test_surrounding_whitespace_is_tolerated():
    assert get_tracker("  kcf  ") is get_tracker("kcf")


def test_unknown_name_raises_value_error_listing_options():
    with pytest.raises(ValueError, match="Unknown tracker type") as excinfo:
        get_tracker("definitely-not-a-tracker")
    assert "csrt" in str(excinfo.value)


def test_never_returns_none():
    """The old implementation silently returned None for unmatched names."""
    for name in ("CSRT", "kcf", "mil"):
        assert get_tracker(name) is not None


def test_legacy_namespace_is_searched(monkeypatch):
    """A tracker present only under cv2.legacy must still resolve."""
    sentinel = object()
    fake_cv2 = SimpleNamespace(
        legacy=SimpleNamespace(TrackerMOSSE_create=sentinel),
        __version__="4.9.0",
    )
    monkeypatch.setattr(tool_box, "cv2", fake_cv2)
    assert tool_box.get_tracker("mosse") is sentinel


def test_main_namespace_wins_over_legacy(monkeypatch):
    main, legacy = object(), object()
    fake_cv2 = SimpleNamespace(
        TrackerCSRT_create=main,
        legacy=SimpleNamespace(TrackerCSRT_create=legacy),
        __version__="4.9.0",
    )
    monkeypatch.setattr(tool_box, "cv2", fake_cv2)
    assert tool_box.get_tracker("csrt") is main


def test_missing_tracker_raises_actionable_runtime_error(monkeypatch):
    fake_cv2 = SimpleNamespace(legacy=SimpleNamespace(), __version__="4.9.0")
    monkeypatch.setattr(tool_box, "cv2", fake_cv2)
    with pytest.raises(RuntimeError) as excinfo:
        tool_box.get_tracker("mosse")
    message = str(excinfo.value)
    assert "mosse" in message
    assert "legacy" in message
    assert "opencv-contrib-python" in message


def test_resolve_passes_callables_through():
    def factory():
        return None

    assert tool_box.resolve_tracker_factory(factory) is factory


def test_available_trackers_is_sorted_and_nonempty():
    names = available_trackers()
    assert names == sorted(names)
    assert "csrt" in names


class TestInspectorVarsValidation:
    """Resolution happens at configuration time, not 30 frames into a video."""

    def test_bad_name_fails_at_construction(self):
        with pytest.raises(ValueError):
            InspectorVars(trck_type="nope")

    def test_default_is_a_plain_string(self):
        assert InspectorVars().trck_type == "CSRT"

    def test_repr_is_readable(self):
        """The default used to render as a raw builtin function pointer."""
        assert "CSRT" in repr(InspectorVars())

    def test_callable_tracker_type_is_accepted(self):
        def factory():
            return None

        assert InspectorVars(trck_type=factory).get_tracker_factory() is factory

    def test_factory_is_cached(self):
        ins = InspectorVars()
        assert ins.get_tracker_factory() is ins.get_tracker_factory()

    def test_reassignment_reresolves(self):
        ins = InspectorVars(trck_type="csrt")
        before = ins.get_tracker_factory()
        ins.trck_type = "kcf"
        assert ins.get_tracker_factory() is not before
