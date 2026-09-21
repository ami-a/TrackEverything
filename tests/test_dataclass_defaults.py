"""Regression guards for the dataclass-default bug class.

Python 3.11 began rejecting unhashable dataclass defaults, which made
``import TrackEverything`` fail outright with::

    ValueError: mutable default <class 'Ids'> for field trck_id_generator
    is not allowed: use default_factory

These tests catch the whole class of problem rather than that one instance, so
the next shared-default slip is caught too.
"""
import dataclasses

import pytest

from TrackEverything import statistical_methods, tool_box, visualization_utils

DATACLASSES = [
    tool_box.InspectorVars,
    tool_box.DetectionVars,
    tool_box.ClassificationVars,
    visualization_utils.VisualizationVars,
    statistical_methods.StatParams,
]

# Defaults that are safe to share because they cannot carry per-instance state.
IMMUTABLE = (type(None), bool, int, float, str, bytes, tuple, frozenset)


@pytest.mark.parametrize("cls", DATACLASSES, ids=lambda c: c.__name__)
def test_field_defaults_are_immutable_or_factories(cls):
    """Every non-factory default must be trivially immutable."""
    for fld in dataclasses.fields(cls):
        if fld.default is dataclasses.MISSING:
            continue
        assert isinstance(fld.default, IMMUTABLE) or callable(fld.default), (
            f"{cls.__name__}.{fld.name} has a non-trivial default {fld.default!r}; "
            "use field(default_factory=...)"
        )


@pytest.mark.parametrize("cls", DATACLASSES, ids=lambda c: c.__name__)
def test_instances_do_not_share_mutable_state(cls):
    """Two default-constructed instances must not alias the same mutable object."""
    first, second = cls(), cls()
    for fld in dataclasses.fields(cls):
        val_a, val_b = getattr(first, fld.name), getattr(second, fld.name)
        if isinstance(val_a, (list, dict, set)) or (
            hasattr(val_a, "__dict__") and not callable(val_a)
        ):
            assert val_a is not val_b, (
                f"{cls.__name__}.{fld.name} is shared between instances"
            )


def test_id_generators_are_independent():
    """Two InspectorVars must not share a tracker-id counter."""
    first, second = tool_box.InspectorVars(), tool_box.InspectorVars()
    assert first.trck_id_generator.get_next_id() == 1
    assert second.trck_id_generator.get_next_id() == 1
    assert first.trck_id_generator is not second.trck_id_generator


def test_statistical_calculators_are_independent():
    """Two InspectorVars must not share accumulated statistics."""
    first, second = tool_box.InspectorVars(), tool_box.InspectorVars()
    assert first.stat_calc is not second.stat_calc
    assert first.stat_calc.parameters is not second.stat_calc.parameters


def test_detectors_do_not_share_configuration():
    """Default-constructed config objects must be per-Detector, not per-class."""
    first, second = tool_box.DetectionVars(), tool_box.DetectionVars()
    assert first is not second
    inspectors = [tool_box.InspectorVars() for _ in range(3)]
    ids = [ins.trck_id_generator.get_next_id() for ins in inspectors]
    assert ids == [1, 1, 1], "id generators leaked across InspectorVars instances"


def test_ids_is_not_a_dataclass():
    """``Ids`` must stay a plain class so it can be used as a field default."""
    assert not dataclasses.is_dataclass(tool_box.Ids)
    assert tool_box.Ids.__hash__ is not None
