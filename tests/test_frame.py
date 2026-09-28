"""Frame resolution and simulation-time frame integration tests.

These tests validate the frame abstraction:
- ``resolve_frame`` converts user-facing frame specs into ``ResolvedFrame``.
- ``simulate`` consumes resolved frames through ``SolveProblem`` state.
"""

from __future__ import annotations


import pytest

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.frames import resolve_frame


pytestmark = pytest.mark.unit


@pytest.fixture
def coupled_chip():
    """Return a coupled transmon-resonator chip plus device handles."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    coupling = Capacitive(q, r, g=0.02)
    chip = Chip([q, r], [coupling])
    return chip, q, r


def test_resolve_frame_float_applies_shared_reference(coupled_chip) -> None:
    """A float frame spec applies the same reference frequency to every device."""
    chip, _, _ = coupled_chip
    resolved = resolve_frame(chip, 5.2)
    assert resolved.mode == "float"
    assert resolved.frequencies == {"q": 5.2, "r": 5.2}


def test_resolve_frame_dict_supports_device_keys_and_missing_defaults(coupled_chip) -> None:
    """A dict frame spec keys by device object and defaults omitted devices to zero."""
    chip, q, _ = coupled_chip
    resolved = resolve_frame(chip, {q: 5.1})
    assert resolved.mode == "dict"
    assert resolved.frequencies["q"] == pytest.approx(5.1)
    assert resolved.frequencies["r"] == pytest.approx(0.0)
