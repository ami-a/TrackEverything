"""The package's public surface and its backwards compatibility."""
import importlib

import pytest

import TrackEverything


def test_version_is_exposed_and_well_formed():
    assert isinstance(TrackEverything.__version__, str)
    parts = TrackEverything.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_every_name_in_all_is_importable():
    for name in TrackEverything.__all__:
        assert hasattr(TrackEverything, name), f"__all__ advertises missing {name}"


def test_star_import_works():
    namespace = {}
    exec("from TrackEverything import *", namespace)
    assert "Detector" in namespace


def test_top_level_imports():
    from TrackEverything import (  # noqa: F401
        ClassificationVars,
        DetectionVars,
        Detector,
        InspectorVars,
        VisualizationVars,
    )


@pytest.mark.parametrize(
    "module",
    [
        "TrackEverything.detector",
        "TrackEverything.inspector",
        "TrackEverything.tool_box",
        "TrackEverything.statistical_methods",
        "TrackEverything.visualization_utils",
    ],
)
def test_legacy_submodule_paths_still_work(module):
    """Existing user code and the example repos import submodules directly."""
    assert importlib.import_module(module) is not None


def test_legacy_detector_import_is_the_same_object():
    from TrackEverything.detector import Detector as Legacy

    assert Legacy is TrackEverything.Detector


def test_package_is_marked_typed():
    from pathlib import Path

    marker = Path(TrackEverything.__file__).parent / "py.typed"
    assert marker.exists(), "py.typed marker is missing from the package"
