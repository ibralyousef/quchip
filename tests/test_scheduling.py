"""Unit tests for QuantumSequence scheduling: cursors, timing, barriers, errors.

Exercises the scheduling API using pure data-manipulation tests:
no simulation engine or backend needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from quchip.chip.chip import Chip
from quchip.control.equipment import ControlEquipment
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Square
from quchip.control.sequence import QuantumSequence
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.ir import evaluate_signal_program


@pytest.fixture
def single_qubit_chip() -> Chip:
    """Single-transmon chip with a ChargeDrive for basic scheduling tests."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    drive = ChargeDrive(target=q0, label="charge_0")
    chip = Chip(devices=[q0])
    chip.connect(ControlEquipment(lines=[drive]))
    return chip


@pytest.fixture
def two_qubit_chip() -> Chip:
    """Two-transmon chip with ChargeDrives for multi-device and barrier tests."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.5, anharmonicity=-0.22, levels=3, label="q1")
    d0 = ChargeDrive(target=q0, label="charge_q0")
    d1 = ChargeDrive(target=q1, label="charge_q1")
    chip = Chip(devices=[q0, q1])
    chip.connect(ControlEquipment(lines=[d0, d1]))
    return chip


@pytest.fixture
def three_qubit_chip() -> Chip:
    """Three-transmon chip with ChargeDrives for selective barrier tests."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.5, anharmonicity=-0.22, levels=3, label="q1")
    q2 = DuffingTransmon(freq=6.0, anharmonicity=-0.20, levels=3, label="q2")
    d0 = ChargeDrive(target=q0, label="charge_q0")
    d1 = ChargeDrive(target=q1, label="charge_q1")
    d2 = ChargeDrive(target=q2, label="charge_q2")
    chip = Chip(devices=[q0, q1, q2])
    chip.connect(ControlEquipment(lines=[d0, d1, d2]))
    return chip


class TestChargeBasic:
    """Basic charge() scheduling and cursor advancement."""


    def test_schedule_phase_rotates_the_resolved_envelope_signal(
        self,
        single_qubit_chip: Chip,
    ) -> None:
        """Scheduling applies global phase after evaluating the local shape."""
        first = QuantumSequence(single_qubit_chip)
        first.charge(
            "q0",
            envelope=Square(duration=10.0, amplitude=0.5),
            freq=5.0,
            phase=0.0,
        )
        second = QuantumSequence(single_qubit_chip)
        second.charge(
            "q0",
            envelope=Square(duration=10.0, amplitude=0.5),
            freq=5.0,
            phase=np.pi / 2.0,
        )

        first_signal = first.resolve().dynamic_terms[0].time_dependence.signal
        second_signal = second.resolve().dynamic_terms[0].time_dependence.signal
        first_value = evaluate_signal_program(first_signal, 0.0, xp=np)
        second_value = evaluate_signal_program(second_signal, 0.0, xp=np)

        assert abs(first_value) == pytest.approx(abs(second_value))
        assert second_value == pytest.approx(1j * first_value)


class TestDelay:
    """delay() advances cursors without scheduling operations."""

    def test_charge_then_delay_then_charge(self, single_qubit_chip: Chip) -> None:
        """delay() between two charge() calls shifts the second op's start time."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.delay("q0", 5.0)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)

        assert seq.scheduled_ops[0].start_time == 0.0
        assert seq.scheduled_ops[1].start_time == 15.0
        assert seq.total_duration == 25.0


class TestExplicitStartTime:
    """schedule() with explicit start_time overrides cursor."""


    def test_start_time_creates_gap(self, single_qubit_chip: Chip) -> None:
        """An explicit start_time beyond the cursor creates an idle gap before the op."""
        q0 = single_qubit_chip.devices[0]
        drive = q0.connected_drives[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.schedule(
            drive,
            envelope=Square(duration=10.0),
            freq=5.0,
            start_time=30.0,
        )

        assert seq.scheduled_ops[0].start_time == 0.0
        assert seq.scheduled_ops[1].start_time == 30.0
        assert seq.total_duration == 40.0


class TestMultiDevice:
    """Multi-device sequences with independent per-device cursors."""

    def test_interleaved_ops_preserve_independence(self, two_qubit_chip: Chip) -> None:
        """Interleaved charge() calls on different devices preserve independent start times."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=20.0), freq=5.5)
        seq.charge("q0", envelope=Square(duration=5.0), freq=5.0)

        assert seq.scheduled_ops[0].start_time == 0.0  # q0 first
        assert seq.scheduled_ops[1].start_time == 0.0  # q1 first (independent)
        assert seq.scheduled_ops[2].start_time == 10.0  # q0 second

        assert seq.channel_cursors[("q0", "charge_q0")] == 15.0
        assert seq.channel_cursors[("q1", "charge_q1")] == 20.0


class TestErrorPaths:
    """Validation and error handling in scheduling."""


    def test_barrier_unknown_device_raises(self, single_qubit_chip: Chip) -> None:
        """barrier() on an unknown device label raises ValueError."""
        seq = QuantumSequence(single_qubit_chip)
        with pytest.raises(ValueError, match="not found on chip"):
            seq.barrier("q_missing")


class TestComplexSequence:
    """Integration-style tests combining multiple operations."""

    def test_charge_barrier_charge_pattern(self, two_qubit_chip: Chip) -> None:
        """Parallel ops followed by barrier() resynchronize both channels before further ops."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=15.0), freq=5.5)
        seq.barrier()
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=10.0), freq=5.5)

        assert seq.scheduled_ops[2].start_time == 20.0
        assert seq.scheduled_ops[3].start_time == 20.0
        assert seq.total_duration == 30.0

    def test_delay_barrier_interaction(self, two_qubit_chip: Chip) -> None:
        """barrier() after a delay on one device pushes the other device's next op up to match."""
        seq = QuantumSequence(two_qubit_chip)
        seq.delay("q0", 50.0)
        seq.barrier()
        seq.charge("q1", envelope=Square(duration=10.0), freq=5.5)

        assert seq.scheduled_ops[0].start_time == 50.0
        assert seq.total_duration == 60.0

    def test_multiple_barriers(self, two_qubit_chip: Chip) -> None:
        """Multiple sequential barrier() calls each resynchronize cursors to the latest."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.barrier()
        seq.charge("q1", envelope=Square(duration=20.0), freq=5.5)
        seq.barrier()
        seq.charge("q0", envelope=Square(duration=5.0), freq=5.0)

        assert seq.scheduled_ops[2].start_time == 30.0
        assert seq.total_duration == 35.0


class TestTotalDuration:
    """total_duration property edge cases."""

    def test_empty_sequence_duration_is_zero(self, single_qubit_chip: Chip) -> None:
        """An empty sequence has total_duration zero."""
        seq = QuantumSequence(single_qubit_chip)
        assert seq.total_duration == 0.0


class TestTracerSafeMax:
    """total_duration and barrier() reduce cursors without a Python max() under jax.jit."""

    def test_total_duration_traced_under_jit(self, two_qubit_chip: Chip) -> None:
        """A traced envelope duration flows through total_duration under jax.jit."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def total_duration(duration):
            seq = QuantumSequence(two_qubit_chip)
            seq.charge("q0", envelope=Square(duration=duration, amplitude=0.5), freq=5.0)
            return seq.total_duration

        result = total_duration(jnp.asarray(20.0))
        assert float(result) == 20.0

    def test_global_barrier_traced_under_jit(self, two_qubit_chip: Chip) -> None:
        """A global barrier() syncs a traced cursor against concrete ones under jax.jit."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def synced_cursors(duration):
            seq = QuantumSequence(two_qubit_chip)
            seq.charge("q0", envelope=Square(duration=duration, amplitude=0.5), freq=5.0)
            seq.charge("q1", envelope=Square(duration=10.0, amplitude=0.5), freq=5.5)
            seq.barrier()
            cursors = seq.channel_cursors
            return cursors[("q0", "charge_q0")], cursors[("q1", "charge_q1")]

        c0, c1 = synced_cursors(jnp.asarray(20.0))
        assert float(c0) == 20.0
        assert float(c1) == 20.0

    def test_selective_barrier_traced_under_jit(self, three_qubit_chip: Chip) -> None:
        """A selective barrier(device) syncs a traced cursor against concrete ones under jax.jit."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def synced_cursors(duration):
            seq = QuantumSequence(three_qubit_chip)
            seq.charge("q0", envelope=Square(duration=duration, amplitude=0.5), freq=5.0)
            seq.charge("q1", envelope=Square(duration=10.0, amplitude=0.5), freq=5.5)
            seq.charge("q2", envelope=Square(duration=3.0, amplitude=0.5), freq=6.0)
            seq.barrier("q0", "q1")
            cursors = seq.channel_cursors
            return cursors[("q0", "charge_q0")], cursors[("q1", "charge_q1")], cursors[("q2", "charge_q2")]

        c0, c1, c2 = synced_cursors(jnp.asarray(20.0))
        assert float(c0) == 20.0
        assert float(c1) == 20.0
        assert float(c2) == 3.0  # q2 was excluded from the barrier group


def test_coarse_output_tlist_preserves_offgrid_pulse(single_qubit_chip: Chip) -> None:
    """A short pulse deep inside a long idle span survives a coarse final-state-only tlist."""
    import numpy as np

    from quchip import Gaussian

    def run(tlist):
        seq = QuantumSequence(single_qubit_chip)
        q = single_qubit_chip.devices[0]
        seq.delay(q, duration=90.0)
        seq.charge(q, envelope=Gaussian(duration=10.0, amplitude=0.05), freq=single_qubit_chip.freq(q))
        return seq.simulate(tlist=tlist)

    dense = run(np.linspace(0.0, 100.0, 2001))
    coarse = run(np.array([0.0, 100.0]))
    p_dense = float(np.asarray(dense.populations[(0,)])[-1])
    p_coarse = float(np.asarray(coarse.populations[(0,)])[-1])
    assert p_dense < 0.99  # the pulse actually did something
    np.testing.assert_allclose(p_coarse, p_dense, atol=5e-3)
