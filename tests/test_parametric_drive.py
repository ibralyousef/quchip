"""ParametricDrive wiring behavior."""

from __future__ import annotations


import pytest

from quchip import (
    Capacitive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    ParametricDrive,
    TunableCapacitive,
)


pytestmark = pytest.mark.unit


def _parts():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    return q0, q1, tc


def test_label_late_binding_via_chip():
    """A ParametricDrive from a coupling label rebinds to the chip's canonical coupling instance once connected."""
    q0, q1, tc = _parts()
    pump = ParametricDrive("tc", label="pump")
    chip = Chip([q0, q1], couplings=[tc])
    chip.connect(ControlEquipment([pump]))
    assert pump._target is tc  # rebound to the canonical instance


def test_static_coupling_is_rejected_with_teaching_error():
    """Wrapping a static coupling in ParametricDrive raises TypeError naming the missing parametric_interaction hook."""
    q0, q1, _ = _parts()
    c = Capacitive(q0, q1, g=0.005, label="c")
    with pytest.raises(TypeError, match="parametric_interaction"):
        ParametricDrive(c)


def test_clone_rebinds_edge_lines():
    """Chip.clone() deep-copies an edge-targeting control line and rebinds it to the clone's own coupling instance."""
    q0, q1, tc = _parts()
    pump = ParametricDrive(tc, label="pump")
    chip = Chip([q0, q1], couplings=[tc], control_equipment=None)
    chip.connect(ControlEquipment([pump]))
    cloned = chip.clone()
    cloned_pump = cloned.control_equipment.lines[0]
    assert cloned_pump is not pump
    assert cloned_pump._target is cloned.coupling("tc")
