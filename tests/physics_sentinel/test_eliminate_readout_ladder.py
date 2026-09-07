"""Retained readout dynamics preserve mapped states, controls and loss operators."""

from __future__ import annotations

from quchip.approximations import Exact

import numpy as np

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    QuantumSequence,
    Resonator,
    Square,
    build_problem,
)
from quchip.chip.transformations import eliminate

_G = 0.05  # qubit-resonator coupling, GHz
_Q_FREQ = 5.0
_R_FREQ = 7.0  # Delta = 2.0 GHz -> g/Delta = 0.025, (g/Delta)^2 = 6.25e-4
_R_LEVELS = 4
_QUALITY_FACTOR = 300.0  # bad-cavity: kappa_ordinary = 7.0/300 = 0.0233 GHz >> |chi|
_AMPLITUDE = 0.005
_DURATION = 30.0  # About two amplitude ring-up times, 2/kappa_angular.


def _readout_chip(*, internal_quality_factor: float | None) -> Chip:
    """Fresh qubit + probed readout resonator, wired with a ``ChargeDrive`` probe on the resonator."""
    q = DuffingTransmon(freq=_Q_FREQ, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=_R_FREQ, levels=_R_LEVELS, internal_quality_factor=internal_quality_factor, label="r")
    readout = ChargeDrive(target=r, label="readout")
    return Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=_G, label="qr")],
        control_equipment=ControlEquipment([readout]),
        frame="lab",
        approximation=Exact(),
    )


def _probe_pointer(chip: Chip, readout_freq: float, initial_state, observable) -> complex:
    """Replay the readout probe on *chip* by its surviving drive label; return the final resonator <a>."""
    tlist = np.linspace(0.0, _DURATION, 81)
    seq = QuantumSequence(chip)
    seq.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    result = seq.simulate(
        tlist=tlist,
        initial_state=initial_state,
        states="final",
    )
    return complex(chip.backend.expect(observable, result.final_state))


def test_readout_pointer_separation_agrees_full_vs_reduced():
    """The transformed drive and observable preserve pointer separation."""
    full = _readout_chip(internal_quality_factor=_QUALITY_FACTOR)
    q, r = full["q"], full["r"]
    readout_freq = 0.5 * (full.freq(r, {q: 0}) + full.freq(r, {q: 1}))
    reduction = eliminate(full, "qr", method="exact")
    reduced, mapping = reduction.chip, reduction.mapping
    observable = full.observable("r", "a")
    states = [full.state(q=level, r=0) for level in (0, 1)]
    a_full_0, a_full_1 = [_probe_pointer(full, readout_freq, state, observable) for state in states]
    a_red_0, a_red_1 = [_probe_pointer(reduced, readout_freq, mapping.project_state(state),
                                      mapping.project_operator(observable)) for state in states]

    sep_full = a_full_1 - a_full_0
    sep_red = a_red_1 - a_red_0

    tol = 3 * (_G / (_R_FREQ - _Q_FREQ)) ** 2
    # The reduction is built from the authored full interaction, so the probe
    # comparison keeps the same non-RWA model on the source side. Both errors
    # remain second order in g/Delta.
    angle_diff = abs(np.angle(sep_full) - np.angle(sep_red))
    assert angle_diff < tol, (angle_diff, tol)
    mag_rel_diff = abs(abs(sep_full) - abs(sep_red)) / abs(sep_full)
    assert mag_rel_diff < tol, (mag_rel_diff, tol)


def test_reduced_readout_does_not_force_density_matrix_solve():
    """A lossless reduction never forces mesolve when the full chip wouldn't need it either."""
    full = _readout_chip(internal_quality_factor=None)
    q, r = full["q"], full["r"]
    readout_freq = 0.5 * (full.freq(r, {q: 0}) + full.freq(r, {q: 1}))
    reduced = eliminate(full, "qr").chip
    tlist = np.linspace(0.0, _DURATION, 81)

    seq_full = QuantumSequence(full)
    seq_full.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    problem_full = build_problem(full, list(seq_full.scheduled_ops), tlist, initial_state=full.state(q=0, r=0))

    seq_reduced = QuantumSequence(reduced)
    seq_reduced.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    problem_reduced = build_problem(
        reduced, list(seq_reduced.scheduled_ops), tlist, initial_state=reduced.state(q=0, r=0)
    )

    assert problem_full.engine_result.collapse_terms == ()
    assert problem_reduced.engine_result.collapse_terms == ()
    chosen_full = problem_full.solver_name(problem_full.chip.backend)
    chosen_reduced = problem_reduced.solver_name(problem_reduced.chip.backend)
    assert chosen_full == chosen_reduced == "sesolve"


def test_reduced_readout_collapse_profile_matches_full_chip():
    """A lossy resonator retains its transformed collapse operator."""
    full = _readout_chip(internal_quality_factor=_QUALITY_FACTOR)
    q, r = full["q"], full["r"]
    readout_freq = 0.5 * (full.freq(r, {q: 0}) + full.freq(r, {q: 1}))
    reduced = eliminate(full, "qr").chip
    tlist = np.linspace(0.0, _DURATION, 81)

    seq_full = QuantumSequence(full)
    seq_full.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    problem_full = build_problem(full, list(seq_full.scheduled_ops), tlist, initial_state=full.state(q=0, r=0))

    seq_reduced = QuantumSequence(reduced)
    seq_reduced.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    problem_reduced = build_problem(
        reduced, list(seq_reduced.scheduled_ops), tlist, initial_state=reduced.state(q=0, r=0)
    )

    assert len(problem_full.engine_result.collapse_terms) == len(problem_reduced.engine_result.collapse_terms) == 1
    mapping = eliminate(full, "qr").mapping
    before = full.backend._collapse_operators(full.resolve())[0]
    after = reduced.backend._collapse_operators(reduced.resolve())[0]
    np.testing.assert_allclose(reduced.backend.to_array(after),
                               reduced.backend.to_array(mapping.project_operator(before)), atol=1e-12)
