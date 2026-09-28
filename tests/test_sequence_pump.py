"""Scheduling edge pumps through QuantumSequence."""

from __future__ import annotations

import pytest

from quchip import (
    Chip, ControlEquipment, DuffingTransmon, ParametricDrive, QuantumSequence, Square, TunableCapacitive,
)


pytestmark = pytest.mark.unit


def _wired_chip():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    pump = ParametricDrive(tc, label="pump")
    chip = Chip([q0, q1], couplings=[tc])
    chip.connect(ControlEquipment([pump]))
    return chip, tc, pump


def test_pump_single_tone_carries_freq_and_phase():
    """Pumping a coupling by its string label carries the supplied carrier frequency and phase into the drive op."""
    chip, tc, _ = _wired_chip()
    seq = QuantumSequence(chip)
    seq.pump("tc", envelope=Square(duration=100.0, amplitude=0.005), freq=0.2, phase=0.3)
    (op,) = seq.scheduled_ops
    assert op.freq == 0.2 and op.phase_offset == 0.3


def test_pump_without_line_raises_with_guidance():
    """Pumping a coupling with no ParametricDrive line wired raises a ValueError that names the missing drive type."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    chip = Chip([q0, q1], couplings=[tc])
    seq = QuantumSequence(chip)
    with pytest.raises(ValueError, match="ParametricDrive"):
        seq.pump(tc, envelope=Square(duration=50.0, amplitude=0.001))
