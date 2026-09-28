"""Unit tests for single-mode band decomposition and demod helpers."""

from __future__ import annotations


import pytest

import numpy as np

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Gaussian
from quchip.control.equipment import ControlEquipment
from quchip.control.sequence import QuantumSequence
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.results import ObservableTrace
from quchip.engine.bands import decompose_bands
from quchip.engine.observables import BandMeta, recombine_expect


def _build_coupled_sequence(frame: str) -> tuple[QuantumSequence, Chip, np.ndarray]:
    """Build a transmon+resonator sequence in a numerically stable regime."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=6.2, levels=8, label="r")
    drive_q = ChargeDrive(target=q, label=f"Dq-{frame}")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=1e-6)],
        control_equipment=ControlEquipment(lines=[drive_q]),
        frame=frame,
    )

    seq = QuantumSequence(chip)
    seq.schedule(
        drive_q,
        envelope=Gaussian(duration=10.0, amplitude=1e-4, sigmas=3),
        freq=q.freq,
    )

    tlist = np.linspace(0.0, 10.0, 501)
    return seq, chip, tlist


_STRICT_SOLVER_OPTS = {
    "nsteps": 500000,
    "atol": 1e-12,
    "rtol": 1e-12,
    "method": "bdf",
}


def test_decompose_bands_completeness() -> None:
    """Sum of all bands reconstructs the original matrix."""
    rng = np.random.default_rng(123)
    mat = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))

    bands = decompose_bands(mat, 5)
    reconstructed = sum(bands.values())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-14)


def test_decompose_bands_supports_jitted_jax_inputs() -> None:
    """Single-mode decomposition should remain usable under ``jax.jit``."""
    import jax
    import jax.numpy as jnp

    matrix = jnp.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [2.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j],
            [0.0 + 0.0j, 4.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    bands = jax.jit(lambda op: decompose_bands(op, 3))(matrix)
    active_weights = {weight for weight, band in bands.items() if np.linalg.norm(np.asarray(band)) > 1e-15}

    assert active_weights == {-1, 1}
    np.testing.assert_allclose(np.asarray(bands[-1]), np.asarray(np.tril(matrix, k=-1)), atol=1e-14)
    np.testing.assert_allclose(np.asarray(bands[1]), np.asarray(np.triu(matrix, k=1)), atol=1e-14)


def test_recombine_remodulate() -> None:
    """Remodulate direction applies exp(-i·ω·w·t) per band (rotating→lab)."""
    tlist = np.array([0.0, 0.125, 0.25])
    freqs = {"q": 5.0, "r": 7.0}

    s_q_p = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    s_q_m = np.array([0.0 + 2.0j, 0.0 + 2.0j, 0.0 + 2.0j])
    s_qr_1 = np.array([0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j])
    s_qr_2 = np.array([0.0 + 0.25j, 0.0 + 0.25j, 0.0 + 0.25j])

    flat_expect = [s_q_p, s_q_m, s_qr_1, s_qr_2]
    meta = [
        BandMeta(key="q", weight=1, device_labels="q"),
        BandMeta(key="q", weight=-1, device_labels="q"),
        BandMeta(key=("q", "r"), weight=(1, -1), device_labels=("q", "r")),
        BandMeta(key=("q", "r"), weight=(-1, 1), device_labels=("q", "r")),
    ]

    band_sum, phase_corrected = recombine_expect(
        flat_expect,
        meta,
        tlist,
        freqs,
        direction="remodulate",
    )

    # Remodulate: exp(-i·ω·w·t)
    phase_q_p = np.exp(-1j * 2 * np.pi * freqs["q"] * (+1) * tlist)
    phase_q_m = np.exp(-1j * 2 * np.pi * freqs["q"] * (-1) * tlist)
    phase_qr_1 = np.exp(-1j * 2 * np.pi * (freqs["q"] * (+1) + freqs["r"] * (-1)) * tlist)
    phase_qr_2 = np.exp(-1j * 2 * np.pi * (freqs["q"] * (-1) + freqs["r"] * (+1)) * tlist)

    expected_q = phase_q_p * s_q_p + phase_q_m * s_q_m
    expected_qr = phase_qr_1 * s_qr_1 + phase_qr_2 * s_qr_2

    # band_sum = direct accumulation
    np.testing.assert_allclose(band_sum["q"], s_q_p + s_q_m, atol=1e-14)
    np.testing.assert_allclose(band_sum[("q", "r")], s_qr_1 + s_qr_2, atol=1e-14)

    # phase_corrected = remodulated (exp(-i·ω·w·t) applied)
    np.testing.assert_allclose(phase_corrected["q"], expected_q, atol=1e-14)
    np.testing.assert_allclose(phase_corrected[("q", "r")], expected_qr, atol=1e-14)


def test_recombine_demodulate() -> None:
    """Demodulate direction applies exp(+i·ω·w·t) per band (lab→slowly varying)."""
    tlist = np.array([0.0, 0.125, 0.25])
    freqs = {"q": 5.0, "r": 7.0}

    s_q_p = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    s_q_m = np.array([0.0 + 2.0j, 0.0 + 2.0j, 0.0 + 2.0j])

    flat_expect = [s_q_p, s_q_m]
    meta = [
        BandMeta(key="q", weight=1, device_labels="q"),
        BandMeta(key="q", weight=-1, device_labels="q"),
    ]

    band_sum, phase_corrected = recombine_expect(
        flat_expect,
        meta,
        tlist,
        freqs,
        direction="demodulate",
    )

    # Demodulate: exp(+i·ω·w·t) — conjugate of remodulate
    phase_q_p = np.exp(+1j * 2 * np.pi * freqs["q"] * (+1) * tlist)
    phase_q_m = np.exp(+1j * 2 * np.pi * freqs["q"] * (-1) * tlist)

    expected_q = phase_q_p * s_q_p + phase_q_m * s_q_m

    np.testing.assert_allclose(band_sum["q"], s_q_p + s_q_m, atol=1e-14)
    np.testing.assert_allclose(phase_corrected["q"], expected_q, atol=1e-14)


@pytest.mark.validation
def test_single_device_demod_matches_lab(backend) -> None:
    """Rotating-frame dict e_ops use identity demodulation (expect == expect_raw)."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")
    seq_rot, chip_rot, _ = _build_coupled_sequence(frame="rotating")

    a_r_lab = chip_lab.device_map["r"].lowering_operator()
    a_r_rot = chip_rot.device_map["r"].lowering_operator()

    init_lab = chip_lab.bare_state(
        q=0,
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.3),
    )
    init_rot = chip_rot.bare_state(
        q=0,
        r=backend.coherent(chip_rot.device_map["r"].levels, 0.3),
    )

    result_lab = seq_lab.simulate(
        tlist=tlist,
        e_ops={"r": a_r_lab},
        initial_state=init_lab,
        options=_STRICT_SOLVER_OPTS,
    )
    result_rot = seq_rot.simulate(
        tlist=tlist,
        e_ops={"r": a_r_rot},
        initial_state=init_rot,
        options=_STRICT_SOLVER_OPTS,
    )

    assert np.max(np.abs(np.asarray(result_lab._expect_data["r"].values))) > 1e-2
    # Rotating frame has demod_freq=0, so demodulation is identity.
    np.testing.assert_allclose(
        np.asarray(result_rot._expect_data["r"].values),
        np.asarray(result_rot._expect_data["r"].raw),
        atol=1e-8,
    )


def test_cross_device_demod(backend) -> None:
    """Tuple-key rotating-frame dict e_ops use identity demodulation."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")
    seq_rot, chip_rot, _ = _build_coupled_sequence(frame="rotating")

    a_q_lab = chip_lab.device_map["q"].lowering_operator()
    a_r_lab = chip_lab.device_map["r"].lowering_operator()

    a_q_rot = chip_rot.device_map["q"].lowering_operator()
    a_r_rot = chip_rot.device_map["r"].lowering_operator()

    init_lab = chip_lab.bare_state(
        q=backend.coherent(chip_lab.device_map["q"].levels, 0.2),
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.2),
    )
    init_rot = chip_rot.bare_state(
        q=backend.coherent(chip_rot.device_map["q"].levels, 0.2),
        r=backend.coherent(chip_rot.device_map["r"].levels, 0.2),
    )

    result_lab = seq_lab.simulate(
        tlist=tlist,
        e_ops={("q", "r"): (a_q_lab, backend.dag(a_r_lab))},
        initial_state=init_lab,
        options=_STRICT_SOLVER_OPTS,
    )
    result_rot = seq_rot.simulate(
        tlist=tlist,
        e_ops={("q", "r"): (a_q_rot, backend.dag(a_r_rot))},
        initial_state=init_rot,
        options=_STRICT_SOLVER_OPTS,
    )

    key = ("q", "r")
    assert key in result_rot._expect_data
    assert isinstance(result_rot._expect_data[key], ObservableTrace)
    assert np.iscomplexobj(np.asarray(result_rot._expect_data[key].values))
    assert np.max(np.abs(np.asarray(result_lab._expect_data[key].values))) > 1e-2

    # Rotating frame has demod_freq=0, so demodulation is identity.
    np.testing.assert_allclose(
        np.asarray(result_rot._expect_data[key].values),
        np.asarray(result_rot._expect_data[key].raw),
        atol=1e-8,
    )


def test_lab_frame_dict_eops(backend) -> None:
    """Lab-frame dict e_ops should expose named traces with raw components."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")

    a_q = chip_lab.device_map["q"].lowering_operator()
    a_r = chip_lab.device_map["r"].lowering_operator()

    dict_e_ops = {
        "r": a_r,
        ("q", "r"): (a_q, backend.dag(a_r)),
    }

    init_state = chip_lab.bare_state(
        q=backend.coherent(chip_lab.device_map["q"].levels, 0.2),
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.2),
    )

    problem = seq_lab.build_problem(
        tlist=tlist,
        e_ops=dict_e_ops,
        initial_state=init_state,
        options=_STRICT_SOLVER_OPTS,
    )
    backend_result = seq_lab._chip.backend.solve_problem(problem)
    flat_expect = (
        list(backend_result.expect.values()) if isinstance(backend_result.expect, dict) else backend_result.expect
    )
    band_sum, phase_corrected = recombine_expect(
        flat_expect=flat_expect,
        meta_list=problem.e_ops_meta or [],
        tlist=problem.tlist,
        frame_freqs=problem.resolved_frame.demod_freqs,
        direction="demodulate",
    )

    result_dict = seq_lab.simulate(
        tlist=tlist,
        e_ops=dict_e_ops,
        initial_state=init_state,
        options=_STRICT_SOLVER_OPTS,
    )

    assert isinstance(result_dict._expect_data, dict)
    assert set(result_dict._expect_data.keys()) == {"r", ("q", "r")}
    assert isinstance(result_dict._expect_data["r"], ObservableTrace)
    assert isinstance(result_dict._expect_data[("q", "r")], ObservableTrace)
    np.testing.assert_allclose(result_dict._expect_data["r"].raw, band_sum["r"], atol=1e-6)
    np.testing.assert_allclose(result_dict._expect_data["r"].values, phase_corrected["r"], atol=1e-6)
    np.testing.assert_allclose(
        result_dict._expect_data[("q", "r")].raw,
        band_sum[("q", "r")],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result_dict._expect_data[("q", "r")].values,
        phase_corrected[("q", "r")],
        atol=1e-6,
    )
