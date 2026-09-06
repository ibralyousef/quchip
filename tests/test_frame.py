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


@pytest.fixture
def coupled_chip():
    """Return a coupled transmon-resonator chip plus device handles."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    coupling = Capacitive(q, r, g=0.02)
    chip = Chip([q, r], [coupling])
    return chip, q, r


def test_resolve_frame_lab_returns_zeros(coupled_chip) -> None:
    """The lab frame resolves to zero reference frequency for every device."""
    chip, _, _ = coupled_chip
    resolved = resolve_frame(chip, "lab")
    assert resolved.mode == "lab"
    assert resolved.frequencies == {"q": 0.0, "r": 0.0}


def test_resolve_frame_rotating_uses_dressed_frequencies(coupled_chip) -> None:
    """The rotating frame resolves each device's reference to its own drive frequency."""
    chip, q, r = coupled_chip
    resolved = resolve_frame(chip, "rotating")
    assert resolved.mode == "rotating"
    assert resolved.frequencies["q"] == pytest.approx(chip.freq(q))
    assert resolved.frequencies["r"] == pytest.approx(chip.freq(r))


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


def test_resolve_frame_rejects_unknown_strings(coupled_chip) -> None:
    """An unrecognized frame string raises ValueError."""
    chip, _, _ = coupled_chip
    with pytest.raises(ValueError, match="Unknown frame string"):
        resolve_frame(chip, "foo")
