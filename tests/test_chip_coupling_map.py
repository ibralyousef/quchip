"""Coupling label access on Chip."""

from __future__ import annotations

import pytest

from quchip import Capacitive, Chip, DuffingTransmon


pytestmark = pytest.mark.unit


def _pair():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    return q0, q1


def test_coupling_map_and_accessor():
    """Coupling map keys couplings by label; coupling() resolves a label or object to that same instance."""
    q0, q1 = _pair()
    c = Capacitive(q0, q1, g=0.005, label="c01")
    chip = Chip([q0, q1], couplings=[c])
    assert chip.coupling_map == {"c01": c}
    assert chip.coupling("c01") is c
    assert chip.coupling(c) is c


def test_device_coupling_label_collision_raises():
    """A coupling label colliding with a device label raises ValueError; couplings and devices share one namespace."""
    q0, q1 = _pair()
    c = Capacitive(q0, q1, g=0.005, label="q0")  # collides with a device label
    with pytest.raises(ValueError, match="q0"):
        Chip([q0, q1], couplings=[c])
