"""Dense crosstalk matrix view on ``ControlEquipment``: extract/perturb/rehydrate/simulate."""

from __future__ import annotations

import numpy as np
import pytest
import quchip

from quchip.chip.chip import Chip
from quchip.control import ChargeDrive, ControlEquipment, Crosstalk
from quchip.control.envelopes import Square
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import Carrier, Constant, DriveOp, Multiply
from quchip.utils.constants import TWO_PI


def _build_two_drive_chip(beta_xt: float = 0.1):
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.5, anharmonicity=-0.2, levels=3, label="q2")
    d1 = ChargeDrive(target=q1, label="d1")
    d2 = ChargeDrive(target=q2, label="d2")
    equip = ControlEquipment(
        lines=[d1, d2],
        signal_chain=[Crosstalk(source=d1.label, victim=d2.label, beta=beta_xt)],
    )
    chip = Chip([q1, q2])
    chip.set_frame("lab")
    chip.connect(equip)
    return chip, equip, d1, d2


def test_crosstalk_matrix_extract_shape_and_labels() -> None:
    """Matrix view reports wiring-order labels and populates the right cell."""
    _, equip, d1, d2 = _build_two_drive_chip(beta_xt=0.1)

    m = equip.crosstalk_matrix()

    assert m.labels == (d1.label, d2.label)
    assert m.beta.shape == (2, 2)
    assert m.theta.shape == (2, 2)
    assert m.delay.shape == (2, 2)
    # Diagonal convention: beta=1, theta=0, delay=0.
    assert m.beta[0, 0] == pytest.approx(1.0)
    assert m.beta[1, 1] == pytest.approx(1.0)
    # Off-diagonal: Crosstalk(source=d1, victim=d2) -> beta[victim=1, source=0]
    assert m.beta[1, 0] == pytest.approx(0.1)
    assert m.beta[0, 1] == pytest.approx(0.0)


def test_crosstalk_delay_transforms_the_complete_carrier_signal() -> None:
    source = quchip.control.AnalyticSignal(
        program=Multiply((Constant(0.3 + 0.2j), Carrier(TWO_PI * 4.7, sign=-1))),
        carrier=4.7,
    )
    transform = Crosstalk(
        source="source",
        victim="victim",
        beta=0.12,
        theta=0.31,
        delay=0.08,
    )

    delivered = transform.apply({("source", 0): source})[("victim", 0)]
    time = 0.43

    expected = 0.12 * np.exp(1j * 0.31) * source.evaluate(time - 0.08, xp=np)
    assert delivered.evaluate(time, xp=np) == pytest.approx(expected)


def test_set_crosstalk_matrix_roundtrip_changes_simulation() -> None:
    """Perturbing a matrix entry and re-injecting it changes simulate() output."""
    chip, equip, d1, d2 = _build_two_drive_chip(beta_xt=0.1)

    envelope = Square(duration=40.0, amplitude=0.1)
    drive_op = DriveOp(
        target_label="q1",
        envelope=envelope,
        freq=5.0,
        start_time=0.0,
        drive_label=d1.label,
    )
    tlist = np.linspace(0.0, 40.0, 201)

    baseline = simulate(chip, [drive_op], tlist)
    p_q2_baseline = baseline.population("q2", 1)

    matrix = equip.crosstalk_matrix()
    perturbed_beta = np.array(matrix.beta, dtype=float, copy=True)
    perturbed_beta[1, 0] = 0.35  # was 0.1

    equip.set_crosstalk_matrix(perturbed_beta)

    perturbed = simulate(chip, [drive_op], tlist)
    p_q2_perturbed = perturbed.population("q2", 1)

    assert np.max(np.abs(p_q2_perturbed - p_q2_baseline)) > 1e-3, (
        "Perturbing the off-diagonal crosstalk amplitude should change "
        "the victim population trajectory."
    )
    edges = equip.crosstalks
    beta_12 = next(
        e.beta for e in edges if e.source == d1.label and e.victim == d2.label
    )
    assert float(beta_12) == pytest.approx(0.35)


def test_unwire_restricts_matrix_to_remaining_lines() -> None:
    """Removing one line preserves the matrix entries among surviving lines."""
    qubits = [
        DuffingTransmon(freq=5.0 + 0.2 * i, anharmonicity=-0.2, levels=2, label=f"q{i}")
        for i in range(3)
    ]
    drives = [ChargeDrive(target=qubit, label=f"d{i}") for i, qubit in enumerate(qubits)]
    equipment = ControlEquipment(drives)
    equipment.set_crosstalk_matrix(np.array([
        [1.0, 0.1, 0.2],
        [0.3, 1.0, 0.4],
        [0.5, 0.6, 1.0],
    ]))
    chip = Chip(qubits, control_equipment=equipment)

    chip.unwire(drives[1])

    reduced = chip.control_equipment.crosstalk_matrix()
    assert reduced.labels == ("d0", "d2")
    np.testing.assert_allclose(reduced.beta, [[1.0, 0.2], [0.5, 1.0]])


def test_crosstalk_matrix_rejects_wrong_shape() -> None:
    """Construction and rebinding reject incompatible crosstalk dimensions."""
    _, equip, *_ = _build_two_drive_chip()
    with pytest.raises(ValueError):
        equip.set_crosstalk_matrix(np.zeros((3, 3)))
    equip.set_crosstalk_matrix(np.zeros((2, 2)))
    for name in ("beta", "theta", "delay"):
        with pytest.raises(ValueError, match="shape"):
            equip.signal_chain[0].with_parameter_value(name, np.zeros((1, 1)))


def test_set_crosstalk_matrix_list_with_traced_entry_is_differentiable():
    import jax
    import jax.numpy as jnp
    from quchip import ChargeDrive, DuffingTransmon
    from quchip.control.equipment import ControlEquipment

    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.26, levels=3)
    q2 = DuffingTransmon(freq=5.2, anharmonicity=-0.26, levels=3)

    def leak(b):
        eq = ControlEquipment([ChargeDrive(target=q1), ChargeDrive(target=q2)])
        eq.set_crosstalk_matrix([[1.0, b], [0.16, 1.0]])
        (matrix,) = eq.signal_chain
        return jnp.sum(matrix.beta)

    assert float(jax.grad(leak)(jnp.asarray(0.14))) == 1.0
