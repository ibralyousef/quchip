"""Convenience API tests for chip accumulator and diagnostics."""

from __future__ import annotations

import pytest

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.signal import Crosstalk, Delay, Gain
from quchip.control.drive import ChargeDrive
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


def _two_device_chip() -> Chip:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = Capacitive(q, r, g=0.02)
    return Chip([q, r], [coupling], label="my_chip")


def test_chip_wire_builds_control_equipment_from_lines() -> None:
    """chip.wire() builds a ControlEquipment from the given lines and signal chain."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], label="wired")

    qubit_drive = ChargeDrive(target=q, label="q")
    readout_drive = ChargeDrive(target=r, label="r")
    equipment = chip.wire(
        qubit_drive,
        readout_drive,
        signal_chain=[Crosstalk(source=qubit_drive.label, victim=readout_drive.label, beta=0.05)],
    )

    assert chip.control_equipment is equipment
    assert [line.label for line in equipment.lines] == ["q", "r"]
    assert len(equipment.signal_chain) == 1
    assert equipment.signal_chain[0].beta == 0.05


def test_chip_status_prints_dashboard_and_returns_none(capsys: pytest.CaptureFixture[str]) -> None:
    """chip.status() prints a dashboard covering label, devices, couplings, frame, and control equipment."""
    chip = _two_device_chip()

    ret = chip.status()
    out = capsys.readouterr().out

    assert ret is None
    assert "my_chip" in out
    assert "q" in out
    assert "r" in out
    assert "Capacitive" in out
    assert "frame" in out.lower()
    assert "dressed" in out.lower()
    assert "control equipment" in out.lower()


@pytest.mark.parametrize(
    "transform",
    [Delay("missing", delta_t=1.0), Crosstalk("missing", "q", beta=0.1), Crosstalk("q", "missing", beta=0.1)],
)
def test_wire_rejects_signal_chain_references_to_missing_lines(transform):
    """Delay targets and both crosstalk endpoints must name wired lines."""
    q = DuffingTransmon(5.0, -0.25, levels=3, label="q")
    drive = ChargeDrive(q, label="q")
    chip = Chip([q])
    chip.wire(drive)
    with pytest.raises(ValueError, match="not in equipment"):
        chip.wire(drive, signal_chain=[transform])


def test_chip_unwire_removes_line_and_chain_references() -> None:
    """Unwiring a line removes it and every signal-chain entry referencing it, leaving unrelated entries."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    q2 = DuffingTransmon(freq=5.3, anharmonicity=-0.24, levels=3, label="q2")
    d1 = ChargeDrive(q, label="d1")
    d2 = ChargeDrive(q2, label="d2")
    chip = Chip([q, q2])
    chip.wire(d1, d2, signal_chain=[Delay("d1", delta_t=1.0), Crosstalk("d1", "d2", beta=0.02), Gain("d2", 2.0)])

    removed = chip.unwire("d1")

    assert removed is d1
    ce = chip.control_equipment
    assert [line.label for line in ce.lines] == ["d2"]
    # Every transform referencing d1 is gone; the untouched Gain on d2 stays.
    assert [type(t).__name__ for t in ce.signal_chain] == ["Gain"]


def test_chip_unwire_accepts_object_and_detaches_last_line() -> None:
    """Unwiring the last remaining line by object clears control_equipment to None."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    d1 = ChargeDrive(q, label="d1")
    chip = Chip([q])
    chip.wire(d1)

    removed = chip.unwire(d1)

    assert removed is d1
    assert chip.control_equipment is None


def test_chip_unwire_unknown_label_raises() -> None:
    """Unwiring a label that is not wired raises ValueError."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])
    chip.wire(ChargeDrive(q, label="d1"))
    with pytest.raises(ValueError, match="No control line labeled 'nope'"):
        chip.unwire("nope")


def test_chip_unwire_without_equipment_raises() -> None:
    """Unwiring on a chip with no control equipment raises ValueError."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])
    with pytest.raises(ValueError, match="nothing is wired"):
        chip.unwire("d1")


def test_chip_disconnect_detaches_and_returns_equipment_for_reconnect() -> None:
    """disconnect() detaches the control equipment and returns it intact for a later connect()."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    d1 = ChargeDrive(q, label="d1")
    chip = Chip([q])
    wired = chip.wire(d1)

    detached = chip.disconnect()

    assert [line.label for line in detached.lines] == [line.label for line in wired.lines]
    assert chip.control_equipment is None

    with pytest.raises(ValueError, match="nothing is wired"):
        chip.disconnect()

    chip.connect(detached)
    assert chip.control_equipment is detached
