"""Tests for lazy auto-dressing and derived Chip quantities (energy, state, freq, frame_info).

System: DuffingTransmon (5.0 GHz, alpha -0.25 GHz, 4 levels) capacitively coupled (g=0.05 GHz)
to a 10-level Resonator (7.0 GHz).
"""

from __future__ import annotations

import pytest

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


OMEGA_Q = 5.0
OMEGA_R = 7.0
ALPHA = -0.25
G = 0.05
Q_LEVELS = 4
R_LEVELS = 10


def _squared_overlap(chip: Chip, left: object, right: object) -> float:
    """Return |⟨left|right⟩|² through the active backend protocol."""
    overlap = chip.backend.overlap(left, right)
    return float(abs(overlap) ** 2)


@pytest.fixture
def dispersive_chip():
    """Build the shared transmon-resonator test system."""
    qubit = DuffingTransmon(
        freq=OMEGA_Q,
        anharmonicity=ALPHA,
        levels=Q_LEVELS,
        label="q",
    )
    resonator = Resonator(freq=OMEGA_R, levels=R_LEVELS, label="r")
    coupling = Capacitive(qubit, resonator, g=G)
    chip = Chip(devices=[qubit, resonator], couplings=[coupling])
    return chip, qubit, resonator


class TestAutoConfig:
    """Tests for auto-dress and derived Chip energy/state APIs."""

    def test_energy_accepts_device_keyed_mapping(self, dispersive_chip) -> None:
        """chip.energy({device: level}) matches the keyword form."""
        chip, qubit, resonator = dispersive_chip

        mapping_value = chip.energy({qubit: 1, resonator: 0})
        keyword_value = chip.energy(q=1, r=0)

        assert mapping_value == pytest.approx(keyword_value)

    def test_energy_invalid_label_raises(self, dispersive_chip) -> None:
        """energy() with invalid state label raises KeyError."""
        chip, _, _ = dispersive_chip
        # q=5 exceeds Q_LEVELS=4, so label won't exist in dressed_eigenvalues
        with pytest.raises(KeyError, match="Available"):
            chip.energy(q=5, r=0)

    def test_state_accepts_device_keyed_mapping(self, dispersive_chip) -> None:
        """chip.state({device: level}) matches the keyword form."""
        chip, qubit, resonator = dispersive_chip
        ds = chip.state({qubit: 0, resonator: 0})
        bare_ground = chip.bare_state({qubit: 0, resonator: 0})
        overlap = _squared_overlap(chip, bare_ground, ds)
        assert overlap > 0.9
