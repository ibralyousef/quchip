"""Parametric capability hooks on couplings."""

from __future__ import annotations

from quchip.approximations import RWA, Exact

import numpy as np

from quchip import Chip, CrossKerr, DuffingTransmon, TunableCapacitive

import pytest


pytestmark = pytest.mark.unit


def _chip(coupling_cls, **kw):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    c = coupling_cls(q0, q1, label="c", **kw)
    return Chip([q0, q1], couplings=[c], approximation=RWA()), c


def _arr(op, *, t=None):
    if hasattr(op, "matrix"):
        return np.asarray(op.matrix(t=t))
    return np.asarray(op.full() if hasattr(op, "full") else op)


def test_tunable_capacitive_drive_is_independent_of_chip_approximation():
    """The coupling authors one interaction before the chip chooses an approximation."""
    from quchip import ParametricDrive
    from quchip.engine.ir import DriveOp
    from quchip.control.envelopes import Square

    chip_rwa, c_rwa = _chip(TunableCapacitive, g_0=0.0)
    chip_full = Chip(
        [d.copy() for d in chip_rwa.devices],
        couplings=[TunableCapacitive("q0", "q1", g_0=0.0, label="c")],
        approximation=Exact(),
    )
    pulse = DriveOp(
        target_label="c",
        drive_label="pump",
        envelope=Square(duration=2.0, amplitude=1.0),
    )
    drive_rwa = ParametricDrive(c_rwa)
    coupling_full = chip_full.coupling("c")
    drive_full = ParametricDrive(coupling_full)
    np.testing.assert_allclose(
        _arr(drive_rwa.hamiltonian(c_rwa, drive_rwa.signal(pulse, c_rwa)), t=1.0),
        _arr(drive_full.hamiltonian(coupling_full, drive_full.signal(pulse, coupling_full)), t=1.0),
    )
    assert not hasattr(c_rwa, "parametric_operator")


def test_crosskerr_interaction_is_diagonal_and_pumpable():
    """CrossKerr's diagonal interaction can also be addressed by a coupling drive."""
    from quchip import ParametricDrive
    from quchip.control.envelopes import Square
    from quchip.engine.ir import DriveOp

    chip, c = _chip(CrossKerr, chi=-0.0005)
    h = _arr(c.interaction_hamiltonian())
    assert np.allclose(h, np.diag(np.diag(h)))  # n̂n̂ is diagonal
    pulse = DriveOp(
        target_label="c",
        drive_label="pump",
        envelope=Square(duration=2.0, amplitude=1.0),
    )
    drive = ParametricDrive(c)
    pumped = _arr(drive.hamiltonian(c, drive.signal(pulse, c)), t=1.0)
    assert np.allclose(pumped, np.diag(np.diag(pumped)))
    assert not hasattr(c, "parametric_operator")


def test_crosskerr_declares_itself():
    """CrossKerr marks itself effective, and its physics_notes disclose the dispersive uniform-pull approximation."""
    chip, c = _chip(CrossKerr, chi=-0.0005)
    assert c.is_effective is True
    notes = " ".join(c.physics_notes()).lower()
    assert "dispersive" in notes and "uniform" in notes
