"""Tests for fit_a_dress: parameter packing, target fitting, and result introspection."""

from __future__ import annotations


import numpy as np
import pytest

from quchip import (
    Exact,
    Capacitive,
    ChargeBasisTransmon,
    Chip,
    CrossKerr,
    DuffingTransmon,
    Resonator,
    TunableCapacitive,
    fit_a_dress,
)
from quchip.chip.coupling_base import BaseCoupling
from quchip.devices.base import BaseDevice
from quchip.inverse_design.fit import _estimate_bare_g, _static_exchange_rate
from quchip.inverse_design import fit as fit_module
from quchip.inverse_design.observables import (
    TargetSpec,
    build_dressed_target_specs,
)
from quchip.inverse_design.subsystems import build_local_subsystem, device_labels_for_local_eval


class _StrengthOnlyCoupling(BaseCoupling):
    """A coupling whose scalar strength lives on ``.strength``, not ``.g``.

    Declares ``coupling_strength_name`` explicitly (unlike the default
    ``"g"``), so this is the general case ``set_coupling_strength`` must
    route through rather than assuming ``.g``.
    """

    _type_prefix = "strength_only"

    def __init__(self, device_a, device_b, *, strength, label=None) -> None:
        super().__init__(device_a, device_b, label=label)
        self.strength = strength

    @property
    def coupling_strength(self) -> float:
        return self.strength

    @property
    def coupling_strength_name(self) -> str:
        return "strength"

    def interaction_hamiltonian(self):
        from typing import cast

        from quchip.backend import get_default_backend

        backend = get_default_backend()
        a = cast(BaseDevice, self.device_a)
        b = cast(BaseDevice, self.device_b)
        return self.strength * backend.tensor(a.number_operator(), b.number_operator())


def test_fit_a_dress_writes_custom_coupling_strength_through_its_own_attribute() -> None:
    """fit_a_dress moves a custom coupling's declared coupling_strength_name attribute, not a stray .g."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = _StrengthOnlyCoupling(q, r, strength=0.01, label="custom")
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(
        chip,
        constraints={coupling: {"cross_kerr": None, "coupling_strength": 0.05}},
        vary={coupling: (coupling.coupling_strength_name,)},
    )

    fitted_coupling = result.chip.couplings[0]
    assert fitted_coupling.strength == pytest.approx(0.05, abs=5e-4)
    assert not hasattr(fitted_coupling, "g")
    assert "custom.strength" in result.final_params
    assert "custom.g" not in result.final_params


def test_fit_a_dress_moves_tunable_capacitive_g0_with_no_stray_g_attribute() -> None:
    """fit_a_dress writes a TunableCapacitive's g_0 (not a stray .g) and reproduces the target."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = TunableCapacitive(q, r, g_0=0.01, label="tc")
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(
        chip,
        constraints={coupling: {"cross_kerr": None, "coupling_strength": 0.03}},
        vary={coupling: (coupling.coupling_strength_name,)},
    )

    fitted_coupling = result.chip.couplings[0]
    assert fitted_coupling.g_0 == pytest.approx(0.03, abs=5e-4)
    assert not hasattr(fitted_coupling, "g")
    assert "tc.g_0" in result.final_params
    assert "tc.g" not in result.final_params


def test_fit_a_dress_moves_crosskerr_chi_with_no_stray_g_attribute() -> None:
    """fit_a_dress writes a CrossKerr's chi (not a stray .g) and reproduces the target."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = CrossKerr(q, r, chi=0.001, label="ck")
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(
        chip,
        constraints={coupling: {"cross_kerr": None, "coupling_strength": 0.003}},
        vary={coupling: (coupling.coupling_strength_name,)},
    )

    fitted_coupling = result.chip.couplings[0]
    assert fitted_coupling.chi == pytest.approx(0.003, abs=5e-4)
    assert not hasattr(fitted_coupling, "g")
    assert "ck.chi" in result.final_params
    assert "ck.g" not in result.final_params


def test_estimate_bare_g_seed_subchip_preserves_chip_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coupling seed sub-chip preserves basis, approximation, and backend intent."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=0.01, label="c")
    chip = Chip(
        [q, r],
        [coupling],
        frame="rotating",
        basis="eigen",
        backend="qutip",
        approximation=Exact(),
    )

    real_chip = fit_module.Chip
    captured: dict = {}

    def spy_chip(devices, couplings=None, **kwargs):
        captured["backend"] = kwargs.get("backend")
        captured["basis"] = kwargs.get("basis")
        captured["approximation"] = kwargs.get("approximation")
        return real_chip(devices, couplings, **kwargs)

    monkeypatch.setattr(fit_module, "Chip", spy_chip)

    _estimate_bare_g(chip, coupling, TargetSpec("cross_kerr", coupling.label, -2e-4))

    assert captured["backend"] is chip.backend
    assert captured["basis"] == "eigen"
    assert captured["approximation"] == Exact()


def test_local_fit_subsystem_inherits_chip_basis_policy() -> None:
    """Local fit evaluation retains an inherited energy-basis projection."""
    q = ChargeBasisTransmon(E_C=0.25, E_J=12.0, num_basis=9, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.02)], basis="eigen")

    local = build_local_subsystem(chip, ("q", "r"))
    resolved = local.resolve(frame="lab")

    assert local.basis == "eigen"
    assert resolved.dims == (3, 4)
    assert resolved.bases["q"].kind == "eigen"


def test_estimate_bare_g_raises_when_target_is_not_bracketed() -> None:
    """_estimate_bare_g raises ValueError (never a saturated endpoint) when the target is unreachable."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=0.001, label="c")
    chip = Chip([q, r], [coupling], frame="rotating")

    huge_target = 1000.0
    with pytest.raises(ValueError, match=r"1000\.0") as exc_info:
        _estimate_bare_g(chip, coupling, TargetSpec("cross_kerr", coupling.label, huge_target))

    message = str(exc_info.value)
    assert "1e-06, 0.25" in message


def test_estimate_bare_g_solves_correct_root_for_a_decreasing_observable(monkeypatch: pytest.MonkeyPatch) -> None:
    """_estimate_bare_g finds the true root even when the observable DECREASES with coupling strength.

    A bisection loop that always assumes "observable increases with
    strength" converges to the wrong endpoint on a decreasing
    observable (it moves the bracket in the wrong direction every
    iteration). The synthetic cross-Kerr below is monotonically
    decreasing on ``seed_strength_bounds`` with a known root, so any
    direction-dependent solver is caught red-handed; a direction-
    independent root solve (``scipy.optimize.brentq``) is not.
    """
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=0.01, label="c")
    chip = Chip([q, r], [coupling], frame="rotating")

    def decreasing_chi(sub_chip, *devices):
        sub_coupling = sub_chip.couplings[0]
        # chi(strength) = 0.5 - strength: root at strength=0.2 for target=0.3,
        # strictly decreasing and strictly positive over (1e-6, 0.25).
        return 0.5 - sub_coupling.coupling_strength

    monkeypatch.setattr(Chip, "static_zz", decreasing_chi)

    seed = _estimate_bare_g(chip, coupling, TargetSpec("cross_kerr", coupling.label, 0.3))

    assert seed == pytest.approx(0.2, abs=1e-8)


def test_fit_a_dress_matches_noncomputational_capacitive_exchange_rate() -> None:
    """The automatic plan fits a resonator pair through dressed exchange."""
    readout = Resonator(freq=7.0, levels=4, label="readout")
    filter_mode = Resonator(freq=7.2, levels=4, label="filter")
    edge = Capacitive(readout, filter_mode, g=0.03, label="readout-filter")

    fit = fit_a_dress(Chip([readout, filter_mode], [edge], frame="rotating"), max_nfev=300)

    report = next(item for item in fit.final_targets if item.label == "readout-filter")
    assert report.kind == "exchange_rate"
    assert report.final == pytest.approx(0.03, abs=1e-8)
    assert float(_static_exchange_rate(fit.chip, ("readout", "filter"))) == pytest.approx(0.03, abs=1e-8)


def test_dressed_target_compilation_never_evaluates_the_desired_chip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The desired chip is a numeric specification, not a runnable seed model."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    edge = Capacitive(q, r, g=-0.00025, label="qr")
    desired = Chip([q, r], [edge], frame="rotating")

    def forbidden(*args, **kwargs):
        raise AssertionError("desired chip was evaluated")

    monkeypatch.setattr(desired, "freq", forbidden)
    monkeypatch.setattr(desired, "dressed_anharmonicity", forbidden)
    monkeypatch.setattr(desired, "static_zz", forbidden)

    specs = build_dressed_target_specs(desired)

    assert [(spec.kind, spec.label, spec.target) for spec in specs] == [
        ("freq", "q", 5.0),
        ("anharmonicity", "q", -0.25),
        ("freq", "r", 7.0),
        ("cross_kerr", "qr", -0.00025),
    ]
    assert [spec.source for spec in specs] == [
        "component default",
        "component default",
        "component default",
        "component default",
    ]


def test_explicit_constraints_extend_replace_and_remove_component_defaults() -> None:
    """Pair constraints are additive; same-edge values replace or remove defaults."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    edge = Capacitive(q0, bus, g=-0.00025, label="q0-bus")
    desired = Chip([q0, q1, bus], [edge], frame="rotating")

    specs = build_dressed_target_specs(
        desired,
        constraints={
            edge: {"cross_kerr": -0.0003},
            (q0, q1): {"exchange_rate": -0.0022},
            (q1, bus): {"zz": 0.00015},
        },
    )
    keyed = {(spec.kind, spec.label): spec.target for spec in specs}

    assert keyed[("cross_kerr", "q0-bus")] == -0.0003
    assert keyed[("exchange_rate", ("q0", "q1"))] == -0.0022
    assert keyed[("cross_kerr", ("q1", "bus"))] == 0.00015
    assert all(
        spec.source == "explicit"
        for spec in specs
        if (spec.kind, spec.label)
        in {
            ("cross_kerr", "q0-bus"),
            ("exchange_rate", ("q0", "q1")),
            ("cross_kerr", ("q1", "bus")),
        }
    )

    without_edge_default = build_dressed_target_specs(
        desired,
        constraints={edge: {"cross_kerr": None}},
    )
    assert ("cross_kerr", "q0-bus") not in {(spec.kind, spec.label) for spec in without_edge_default}


def test_fit_a_dress_treats_a_capacitive_scalar_as_a_cross_kerr_target() -> None:
    """The desired edge number is a dressed constraint, not the fitted bare g."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    edge = Capacitive(q, r, g=-0.00025, label="qr")
    desired = Chip([q, r], [edge], frame="rotating")

    fit = fit_a_dress(desired, max_nfev=300)

    assert fit.chip.static_zz("q", "r") == pytest.approx(-0.00025, abs=2e-6)
    assert fit.final_params["qr.g"] > 0.0
    assert fit.solver_info["n_free_parameters"] == 4
    assert fit.solver_info["n_target_residuals"] == 4
    assert fit.solver_info["input_contract"] == "desired-chip"
    assert {(report.kind, report.label): report.source for report in fit.final_targets}[
        ("cross_kerr", "qr")
    ] == "component default"
    coupling_report = next(report for report in fit.parameter_reports if report.name == "qr.g")
    assert coupling_report.seed_source == "isolated-pair root solve"
    assert coupling_report.sign_choice == "positive convention"
    assert coupling_report.lower_bound < 0.0 < coupling_report.upper_bound


def test_fit_a_dress_records_a_user_supplied_coupling_sign() -> None:
    """An explicit coupling start owns the sign branch and the receipt says so."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    edge = Capacitive(q, r, g=-0.00025, label="qr")
    desired = Chip([q, r], [edge], frame="rotating")

    fit = fit_a_dress(desired, start={"qr.g": -0.05})

    report = next(report for report in fit.parameter_reports if report.name == "qr.g")
    assert report.initial == pytest.approx(-0.05)
    assert report.final < 0.0
    assert report.seed_source == "user start"
    assert report.sign_choice == "user supplied"


def test_automatic_desired_chip_fit_rejects_a_rank_deficient_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal target/parameter counts do not make an automatic flat direction safe."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q], frame="rotating")

    def only_freq(candidate, spec, evaluator):
        del spec, evaluator
        return candidate["q"].freq

    monkeypatch.setattr(fit_module, "_evaluate_spec", only_freq)

    with pytest.raises(ValueError, match=r"Jacobian rank 1 for 2 free parameters"):
        fit_a_dress(desired)


def test_stopped_automatic_fit_retains_rank_deficient_candidate(monkeypatch):
    desired = Chip([DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")])
    monkeypatch.setattr(fit_module, "_evaluate_spec", lambda candidate, spec, evaluator: candidate["q"].freq)

    with pytest.warns(UserWarning, match="Jacobian rank 1 for 2 free parameters"):
        fit = fit_a_dress(desired, max_nfev=1)

    assert fit.converged is False
    assert "maximum" in fit.message.lower()
    assert fit.chip["q"].freq == pytest.approx(5.0)
    assert fit.history[-1] == pytest.approx(fit.loss)
    assert fit.solver_info["jacobian_rank"] == 1


def test_manual_rank_deficient_fit_returns_diagnostics_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit vary plan may return ambiguity, but it cannot hide it."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q], frame="rotating")

    def only_freq(candidate, spec, evaluator):
        del spec, evaluator
        return candidate["q"].freq

    monkeypatch.setattr(fit_module, "_evaluate_spec", only_freq)

    with pytest.warns(UserWarning, match=r"Jacobian rank 1 for 2 free parameters"):
        fit = fit_a_dress(
            desired,
            vary={q: ("freq", "anharmonicity")},
        )

    assert fit.solver_info["jacobian_rank"] == 1
    assert fit.solver_info["rank_deficient"] is True
    assert np.isinf(fit.solver_info["jacobian_condition_number"])
    weak = fit.solver_info["weak_parameter_directions"]
    assert weak
    assert abs(weak[0]["relative_weights"]["q.anharmonicity"]) == pytest.approx(1.0)


def test_automatic_desired_chip_fit_rejects_an_underdetermined_target_plan() -> None:
    """Removing a default target cannot leave an automatic free parameter floating."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q], frame="rotating")

    with pytest.raises(ValueError, match=r"2 free parameters but only 1 target residual"):
        fit_a_dress(desired, constraints={q: {"anharmonicity": None}})


def test_fit_a_dress_accepts_manual_vary_and_start_overrides() -> None:
    """Advanced users can replace automatic parameter selection and starting values."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q], frame="rotating")

    fit = fit_a_dress(
        desired,
        vary={q: ("freq",)},
        start={"q.freq": 4.8},
    )

    assert fit.initial_params == {"q.freq": 4.8}
    assert fit.final_params["q.freq"] == pytest.approx(5.0, abs=1e-8)
    assert fit.chip.dressed_anharmonicity("q") == pytest.approx(-0.25, abs=1e-8)


def test_selection_errors_identify_vary() -> None:
    """Desired-chip errors use the vocabulary shown in its public signature."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q])

    with pytest.raises(ValueError, match=r"vary\['q'\]"):
        fit_a_dress(desired, vary={q: "freq"})


def test_fit_a_dress_retains_scipy_jacobian_without_dynamiqs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A QuTiP-only installation retains SciPy's numerical Jacobian."""

    def unavailable(*args, **kwargs):
        raise ImportError("dynamiqs unavailable")

    monkeypatch.setattr(fit_module, "_jax_residual_functions", unavailable)
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating")

    result = fit_a_dress(chip)

    assert result.solver_info["jacobian"] == "finite-difference"


def test_fit_a_dress_respects_a_qutip_chip_backend() -> None:
    """A QuTiP chip retains SciPy's numerical Jacobian when dynamiqs is installed."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", backend="qutip")

    result = fit_a_dress(chip)

    assert result.solver_info["jacobian"] == "finite-difference"


def test_fit_a_dress_recovers_qr_target_chi_from_declared_coupling_value() -> None:
    """fit_a_dress recovers qubit-resonator chi from a coupling's declared g."""
    q = DuffingTransmon(freq=5.241031326, anharmonicity=-0.261031326, levels=4, label="q")
    r = Resonator(freq=6.653024480, levels=10, label="r")
    coupling = Capacitive(q, r, g=-646019e-9)
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(chip, constraints={coupling: {"cross_kerr": 2 * coupling.g}}, max_hilbert_dim=10_000)

    assert result.chip is not chip
    fitted_chip = result.chip
    fitted_q = fitted_chip["q"]
    fitted_r = fitted_chip["r"]
    fitted_c = fitted_chip.couplings[0]

    chi = (fitted_chip.freq(fitted_r, when={fitted_q: 1}) - fitted_chip.freq(fitted_r, when={fitted_q: 0})) / 2.0
    assert fitted_chip.freq(fitted_q) == pytest.approx(5.241031326, abs=5e-4)
    assert fitted_chip.freq(fitted_r) == pytest.approx(6.653024480, abs=5e-4)
    assert fitted_chip.dressed_anharmonicity(fitted_q) == pytest.approx(-0.261031326, abs=5e-4)
    assert chi == pytest.approx(-646019e-9, abs=5e-6)
    assert np.isfinite(fitted_c.g)


def test_fit_a_dress_recovers_qq_target_zz_from_declared_coupling_value() -> None:
    """fit_a_dress recovers static ZZ between two qubits from a coupling's declared g."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.18, anharmonicity=-0.24, levels=3, label="q1")
    coupling = Capacitive(q0, q1, g=0.0015)
    chip = Chip([q0, q1], [coupling], frame="rotating")

    result = fit_a_dress(chip, max_hilbert_dim=10_000)

    fitted_chip = result.chip
    fitted_q0 = fitted_chip["q0"]
    fitted_q1 = fitted_chip["q1"]
    assert fitted_chip.freq(fitted_q0) == pytest.approx(5.0, abs=5e-4)
    assert fitted_chip.freq(fitted_q1) == pytest.approx(5.18, abs=5e-4)
    assert fitted_chip.static_zz(fitted_q0, fitted_q1) == pytest.approx(0.0015, abs=5e-5)


def test_fit_a_dress_does_not_mutate_input_chip() -> None:
    """fit_a_dress leaves the input chip's device and coupling parameters unmutated."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=-1.2e-4)
    chip = Chip([q, r], [coupling], frame="rotating")

    original = (q.freq, q.anharmonicity, coupling.g)
    _ = fit_a_dress(chip)

    assert (q.freq, q.anharmonicity, coupling.g) == original


def test_fit_a_dress_respects_coupling_target_override_to_g() -> None:
    """fit_a_dress fits a coupling's raw g directly when the target kind is 'g'."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.04)], frame="rotating")

    result = fit_a_dress(
        chip,
        constraints={chip.couplings[0]: {"cross_kerr": None, "coupling_strength": 0.04}},
        vary={chip.couplings[0]: ("g",)},
    )

    assert result.final_params[f"{chip.couplings[0].label}.g"] == pytest.approx(0.04, abs=5e-4)


def test_fit_a_dress_requires_explicit_local_evaluation() -> None:
    """A resource ceiling stops the full fit; local physics requires selection."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    r0 = Resonator(freq=7.0, levels=3, label="r0")
    r1 = Resonator(freq=7.3, levels=3, label="r1")
    chip = Chip(
        [q0, q1, r0, r1],
        [
            Capacitive(q0, r0, g=-1.0e-4),
            Capacitive(q1, r1, g=-1.2e-4),
            Capacitive(q0, q1, g=0.001),
        ],
        frame="rotating",
    )

    with pytest.raises(ValueError, match="dimension.*81.*50"):
        fit_a_dress(chip, max_hilbert_dim=50)
    result = fit_a_dress(chip, evaluator="local")

    assert any(report.evaluator == "local" for report in result.final_targets)


@pytest.mark.parametrize("evaluator", ["auto", "bad", None])
def test_fit_rejects_unknown_evaluator(evaluator):
    chip = Chip([Resonator(freq=5.0, levels=2, label="r")])
    with pytest.raises(ValueError, match="evaluator"):
        fit_a_dress(chip, evaluator=evaluator)


def test_local_fit_limit_applies_to_actual_neighborhood():
    modes = [Resonator(freq=5.0 + i, levels=3, label=f"r{i}") for i in range(4)]
    chip = Chip(modes)
    local = fit_a_dress(chip, evaluator="local", max_hilbert_dim=3)
    assert all(report.evaluator == "local" for report in local.final_targets)
    with pytest.raises(ValueError, match="dimension.*3.*2"):
        fit_a_dress(chip, evaluator="local", max_hilbert_dim=2)


def test_fit_dimension_limit_cannot_overflow_for_many_devices():
    chip = Chip([Resonator(freq=5.0, levels=2, label=f"r{i}") for i in range(64)])
    assert chip.total_dim == 2**64
    with pytest.raises(ValueError, match=str(2**64)):
        fit_a_dress(chip)


def test_fit_a_dress_history_records_objective_evaluations() -> None:
    """The returned history supports a real convergence plot, not two endpoints."""
    q = DuffingTransmon(freq=4.8, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", backend="qutip")

    result = fit_a_dress(
        chip,
        constraints={q: {"freq": 5.1}},
        vary={q: ("freq",)},
    )

    assert result.history.ndim == 1
    assert len(result.history) > 2
    assert result.history[0] > result.history[-1]
    assert result.history[-1] == pytest.approx(result.loss)
    assert result.solver_info["history_axis"] == "distinct residual evaluation"
    assert result.solver_info["n_recorded_evaluations"] == len(result.history)


def test_fit_retains_nonconverged_candidate_and_explains_status():
    from dataclasses import replace

    desired = Chip([DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q")])
    fit = fit_a_dress(desired, vary={"q": ("freq",)}, start={"q.freq": 4.8}, max_nfev=1)
    assert fit.converged is False
    assert "maximum" in fit.message.lower()
    assert fit.chip["q"].freq == pytest.approx(4.8)
    assert fit.loss > 0
    assert fit.history[-1] == pytest.approx(fit.loss)
    unknown = replace(fit, solver_info={})
    assert unknown.converged is None
    assert unknown.message is None


def test_fit_rebind_returns_fitted_clones_for_seed_devices() -> None:
    """fit.rebind(*seeds) shortcircuits the ``chip.device_map[qb.label]`` ritual."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=-1.2e-4)], frame="rotating")

    result = fit_a_dress(chip)

    q_f, r_f = result.rebind(q, r)
    assert q_f is result.chip.device_map["q"]
    assert r_f is result.chip.device_map["r"]

    assert result.rebind(q) is result.chip.device_map["q"]
    assert result.rebind("r") is result.chip.device_map["r"]

    assert q_f is not q
    assert r_f is not r

    import pytest as _pytest

    with _pytest.raises(ValueError):
        result.rebind()


def test_fit_a_dress_accepts_constraints_with_object_labels() -> None:
    """fit_a_dress accepts device objects as constraint keys."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=-1.2e-4)
    chip = Chip([q, r], [coupling], frame="rotating")

    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(
            chip,
            vary={q: ("freq", "anharmonicity"), r: ("freq",), coupling: ("g",)},
            constraints={q: {"freq": 5.0}, r: {"freq": 7.0}, coupling: {"cross_kerr": None}},
        )

    assert not any(report.kind in {"cross_kerr", "coupling_strength"} for report in result.final_targets)
    assert any(report.kind == "freq" and report.label == "q" for report in result.final_targets)
    assert any(report.kind == "freq" and report.label == "r" for report in result.final_targets)


def test_fit_a_dress_recovers_static_exchange_for_sheldon_style_bus_model() -> None:
    """fit_a_dress recovers a targeted static exchange coupling in a bus-mediated three-device system."""
    control = DuffingTransmon(freq=5.08, anharmonicity=-0.31, levels=4, label="control")
    target = DuffingTransmon(freq=4.95, anharmonicity=-0.35, levels=4, label="target")
    bus = Resonator(freq=6.28, levels=6, label="bus")
    c_bus = Capacitive(control, bus, g=0.020, label="c_bus")
    t_bus = Capacitive(target, bus, g=0.017, label="t_bus")
    chip = Chip([control, target, bus], [c_bus, t_bus], frame="rotating")

    # 7 free bare parameters (control freq/anharmonicity, target freq/anharmonicity, bus
    # freq, c_bus.g, t_bus.g) against 6 target residuals: underdetermined by count, yet
    # the fit converges because the exchange target and the two per-device anchors jointly
    # pin the coupling split closely enough from these seeds.
    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(
            chip,
            vary={
                control: ("freq", "anharmonicity"),
                target: ("freq", "anharmonicity"),
                bus: ("freq",),
                c_bus: ("g",),
                t_bus: ("g",),
            },
            constraints={
                c_bus: {"cross_kerr": None},
                t_bus: {"cross_kerr": None},
                control: {"freq": 5.114, "anharmonicity": -0.330},
                target: {"freq": 4.914, "anharmonicity": -0.330},
                bus: {"freq": 6.31},
                (control, target): {"exchange": 0.0038},
            },
            max_hilbert_dim=1_000,
        )

    fitted_chip = result.chip
    fitted_control = fitted_chip["control"]
    fitted_target = fitted_chip["target"]
    fitted_bus = fitted_chip["bus"]
    exchange_h = fitted_chip.effective_subspace_hamiltonian(
        ({fitted_control: 1, fitted_target: 0, fitted_bus: 0}, {fitted_control: 0, fitted_target: 1, fitted_bus: 0})
    )

    assert fitted_chip.freq(fitted_control) == pytest.approx(5.114, abs=1e-3)
    assert fitted_chip.freq(fitted_target) == pytest.approx(4.914, abs=1e-3)
    assert fitted_chip.freq(fitted_bus) == pytest.approx(6.31, abs=1e-3)
    assert fitted_chip.dressed_anharmonicity(fitted_control) == pytest.approx(-0.330, abs=2e-3)
    assert fitted_chip.dressed_anharmonicity(fitted_target) == pytest.approx(-0.330, abs=2e-3)
    assert exchange_h[0, 1] == pytest.approx(0.0038, abs=2e-4)


def test_fit_a_dress_respects_signed_exchange_target_for_direct_qq_system() -> None:
    """fit_a_dress preserves the sign of a targeted direct qubit-qubit exchange coupling."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.18, anharmonicity=-0.24, levels=3, label="q1")
    coupling = Capacitive(q0, q1, g=-0.0015, label="qq")
    chip = Chip([q0, q1], [coupling], frame="rotating")

    result = fit_a_dress(
        chip,
        vary={q0: ("freq", "anharmonicity"), q1: ("freq", "anharmonicity"), coupling: ("g",)},
        constraints={
            coupling: {"cross_kerr": None},
            q0: {"freq": 5.0, "anharmonicity": -0.25},
            q1: {"freq": 5.18, "anharmonicity": -0.24},
            (q0, q1): {"exchange": -0.0015},
        },
    )

    fitted_q0 = result.chip["q0"]
    fitted_q1 = result.chip["q1"]
    exchange_h = result.chip.effective_subspace_hamiltonian(
        ({fitted_q0: 1, fitted_q1: 0}, {fitted_q0: 0, fitted_q1: 1})
    )
    assert exchange_h[0, 1] == pytest.approx(-0.0015, abs=5e-5)


def test_fit_a_dress_recovers_explicit_pair_zz_target_for_direct_qq_system() -> None:
    """fit_a_dress recovers an explicit pair-level zz target for a direct qubit-qubit system."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.18, anharmonicity=-0.24, levels=3, label="q1")
    coupling = Capacitive(q0, q1, g=0.0015, label="qq")
    chip = Chip([q0, q1], [coupling], frame="rotating")

    result = fit_a_dress(
        chip,
        vary={q0: ("freq", "anharmonicity"), q1: ("freq", "anharmonicity"), coupling: ("g",)},
        constraints={
            coupling: {"cross_kerr": None},
            q0: {"freq": 5.0, "anharmonicity": -0.25},
            q1: {"freq": 5.18, "anharmonicity": -0.24},
            (q0, q1): {"zz": 0.0015},
        },
    )

    fitted_q0 = result.chip["q0"]
    fitted_q1 = result.chip["q1"]
    assert result.chip.static_zz(fitted_q0, fitted_q1) == pytest.approx(0.0015, abs=5e-5)


def test_device_labels_for_local_eval_stays_one_hop_for_pair_targets() -> None:
    """device_labels_for_local_eval expands only one hop beyond single or pair targets."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q2")
    q3 = DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q3")
    chip = Chip(
        [q0, q1, q2, q3],
        [Capacitive(q0, q1, g=0.001), Capacitive(q1, q2, g=0.001), Capacitive(q2, q3, g=0.001)],
        frame="rotating",
    )

    assert device_labels_for_local_eval(chip, "q1") == ("q0", "q1", "q2")
    assert device_labels_for_local_eval(chip, ("q1", "q2")) == ("q0", "q1", "q2", "q3")
