"""Tests for label resolution and label-keyed dict lookup."""

from __future__ import annotations

import pytest

from quchip.utils.labeling import (
    bare_label_from_mapping,
    merge_labeled_values,
)


pytestmark = pytest.mark.unit


class _FakeDevice:
    """Minimal label-bearing stand-in for a device/coupling/drive object."""

    def __init__(self, label: str) -> None:
        self.label = label


def test_merge_labeled_values_raises_on_duplicate_mapping_keys():
    """Two mapping keys resolving to the same label raise ValueError."""
    dev = _FakeDevice("q0")
    with pytest.raises(ValueError, match="q0"):
        merge_labeled_values({dev: 1, "q0": 2}, {})


def test_merge_labeled_values_raises_on_mapping_kwargs_collision():
    """A label present in both the mapping and kwargs raises ValueError."""
    with pytest.raises(ValueError, match="q0"):
        merge_labeled_values({"q0": 1}, {"q0": 2})


def test_bare_label_from_mapping_fills_unmentioned_devices_with_zero():
    """Devices absent from the spec default to Fock index 0."""
    result = bare_label_from_mapping(["q0", "q1", "q2"], {"q1": 3}, {})
    assert result == (0, 3, 0)
