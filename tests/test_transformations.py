"""Tests for the model-reduction transform ``eliminate()`` and the ``ChipTransform`` protocol."""

import numpy as np
import pytest

from quchip import Capacitive, Chip, CrossKerr, DuffingTransmon, Resonator, TunableCapacitive


def test_eliminate_resonator_retains_lamb_shift_and_purcell():
    """Eliminating a resonator retains its Lamb shift and Purcell decay separately from authored parameters."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, internal_quality_factor=5000.0, levels=4, label="r")
    g = 0.08
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
    from quchip.chip.transformations import eliminate

    res = eliminate(chip, "r")
    reduced = res.chip
    assert [d.label for d in reduced.devices] == ["q"]

    delta = 5.0 - 7.0
    lamb = g**2 / delta
    kappa = 2 * np.pi * 7.0 / 5000.0
    purcell = np.sin(g / delta) ** 2 * kappa

    assert reduced["q"].freq == 5.0
    assert reduced.freq("q") == pytest.approx(5.0 + lamb, rel=1e-6)
    jumps = reduced.backend._collapse_operators(reduced.resolve(frame="lab"))
    assert sum(abs(reduced.backend.to_array(op)[0, 1])**2 for op in jumps) == pytest.approx(purcell, rel=1e-6)
    assert reduced["q"].T1 is None
    assert res.validity["cap_0"]["g_over_delta"] == pytest.approx(abs(g / delta), rel=1e-6)


def _bridge_chip(direct_g=None):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    couplings = [Capacitive(q0, bus, g=0.05, label="leg0"), Capacitive(q1, bus, g=0.05, label="leg1")]
    if direct_g is not None:
        couplings.append(Capacitive(q0, q1, g=direct_g, label="direct"))
    return Chip([q0, q1, bus], couplings=couplings)


def test_eliminate_bridge_derives_mediated_exchange():
    """Eliminating a bridging bus derives the mediated exchange J = (g1 g2/2)(1/Δ1 + 1/Δ2)."""
    from quchip.chip.transformations import eliminate

    res = eliminate(_bridge_chip(), "bus")
    reduced = res.chip

    j_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    assert [d.label for d in reduced.devices] == ["q0", "q1"]
    (mediated,) = reduced.couplings
    assert type(mediated) is Capacitive
    assert mediated.label == "elim_bus"
    assert float(mediated.g) == pytest.approx(j_expected, rel=1e-9)

    assert res.effective_params["q0"]["freq_after"] == pytest.approx(5.0 + 0.05**2 / (5.0 - 7.0), rel=1e-9)
    assert res.effective_params["q1"]["freq_after"] == pytest.approx(5.1 + 0.05**2 / (5.1 - 7.0), rel=1e-9)
    assert set(res.validity) == {"leg0", "leg1"}
    assert res.effective_params["exchange"]["between"] == ("q0", "q1")


def test_eliminate_bridge_can_cancel_an_existing_direct_interaction():
    """Direct and mediated exchanges cancel in the complete reduced Hamiltonian."""
    from quchip.chip.transformations import eliminate

    j_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    res = eliminate(_bridge_chip(direct_g=-j_expected), "bus")

    direct = res.chip.coupling("direct")
    assert float(direct.g) == pytest.approx(-j_expected, abs=1e-12)
    assert res.effective_params["exchange"]["coupling"] == "elim_bus"
    h = res.chip.hamiltonian().matrix()
    assert complex(h[2, 1]) == pytest.approx(0.0, abs=1e-12)


def test_eliminate_bridge_preserves_non_foldable_direct_edge_without_double_counting():
    """A direct edge that owns its authored interaction is preserved unchanged; no double-counted exchange."""
    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class CustomExchange(CouplingModel):
        """Minimal honest exchange coupling that owns its authored interaction."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b, p):
            return p.j * (a.adag * b.a + a.a * b.adag)

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    direct_j = 0.01
    chip = Chip(
        [q0, q1, bus],
        couplings=[
            Capacitive(q0, bus, g=0.05, label="leg0"),
            Capacitive(q1, bus, g=0.05, label="leg1"),
            CustomExchange(q0, q1, j=direct_j, label="direct"),
        ],
    )

    res = eliminate(chip, "bus")
    reduced = res.chip

    assert {c.label for c in reduced.couplings} == {"direct", "elim_bus"}
    assert float(reduced.coupling_map["direct"].j) == pytest.approx(direct_j)  # untouched

    j_mediated_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    parallel_edge = reduced.coupling_map["elim_bus"]
    assert float(parallel_edge.g) == pytest.approx(j_mediated_expected, rel=1e-6)

    # The reduced chip's own total exchange (read the same way the fold
    # itself measures it) must equal direct_j + j_mediated exactly once —
    # a double count would read 2*direct_j + j_mediated instead.
    h, labels, dims = bare_hamiltonian(reduced)
    row = bare_index(labels, dims, "q0")
    col = bare_index(labels, dims, "q1")
    assert complex(h[row, col]).real == pytest.approx(direct_j + j_mediated_expected, rel=1e-6)


def test_eliminate_bridge_direct_exchange_is_counted_once():
    """An authored direct exchange edge contributes once to the reduced fold."""
    import warnings

    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class DirectExchange(CouplingModel):
        """Exchange-only authored interaction."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b, p):
            return p.j * (a.adag * b.a + a.a * b.adag)

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    chip = Chip(
        [q0, q1, bus],
        couplings=[
            Capacitive(q0, bus, g=0.05, label="leg0"),
            Capacitive(q1, bus, g=0.05, label="leg1"),
            DirectExchange(q0, q1, j=0.01, label="direct"),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = eliminate(chip, "bus")
        reduced = res.chip

        j_mediated_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
        parallel_edge = reduced.coupling_map["elim_bus"]
        assert float(parallel_edge.g) == pytest.approx(j_mediated_expected, rel=1e-6)

        h, labels, dims = bare_hamiltonian(reduced)
        row = bare_index(labels, dims, "q0")
        col = bare_index(labels, dims, "q1")
        assert complex(h[row, col]).real == pytest.approx(0.01 + j_mediated_expected, rel=1e-6)


def test_eliminate_bridge_fold_target_and_preserved_edge_are_each_counted_exactly_once():
    """A foldable edge and a preserved edge between the same pair each contribute exactly once."""
    from quchip.chip.sw import bare_hamiltonian, bare_index
    from quchip.chip.transformations import eliminate
    from quchip.declarative.models import CouplingModel
    from quchip.declarative.parameters import Scalar, parameter

    class CustomExchange(CouplingModel):
        """Minimal honest exchange coupling that owns its authored interaction."""

        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b, p):
            return p.j * (a.adag * b.a + a.a * b.adag)

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    direct_j = 0.01
    foldable_g = 0.02
    chip = Chip(
        [q0, q1, bus],
        couplings=[
            Capacitive(q0, bus, g=0.05, label="leg0"),
            Capacitive(q1, bus, g=0.05, label="leg1"),
            Capacitive(q0, q1, g=foldable_g, label="foldable"),
            CustomExchange(q0, q1, j=direct_j, label="preserved"),
        ],
    )

    res = eliminate(chip, "bus")
    reduced = res.chip

    assert {c.label for c in reduced.couplings} == {"foldable", "preserved", "elim_bus"}
    assert float(reduced.coupling_map["preserved"].j) == pytest.approx(direct_j)  # untouched

    j_mediated_expected = 0.05 * 0.05 / 2.0 * (1.0 / (5.0 - 7.0) + 1.0 / (5.1 - 7.0))
    total_expected = foldable_g + direct_j + j_mediated_expected

    h, labels, dims = bare_hamiltonian(reduced)
    row = bare_index(labels, dims, "q0")
    col = bare_index(labels, dims, "q1")
    assert complex(h[row, col]).real == pytest.approx(total_expected, rel=1e-6)


def test_eliminate_bridge_purcell_from_mode_t1():
    """A dissipative bridge carries its collective bright-state decay through the retained rotation."""
    from quchip.chip.transformations import eliminate

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus", T1=10_000.0)
    chip = Chip([q0, q1, bus], couplings=[Capacitive(q0, bus, g=0.05), Capacitive(q1, bus, g=0.05)])

    res = eliminate(chip, "bus")
    angles = np.array([0.05 / (5.0 - 7.0), 0.05 / (5.1 - 7.0)])
    expected_rate = (angles[0] * np.sinc(np.linalg.norm(angles) / np.pi)) ** 2 / 10_000.0
    jumps = res.chip.backend._collapse_operators(res.chip.resolve(frame="lab"))
    psi = res.chip.backend.to_array(res.chip.bare_state(q0=1))
    assert sum(np.linalg.norm(res.chip.backend.to_array(op) @ psi)**2 for op in jumps) == pytest.approx(
        expected_rate, rel=1e-9)


def test_eliminate_purcell_keeps_intrinsic_thermal_noise_separate():
    """Inherited emission and intrinsic thermal absorption remain distinct transformed channels."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(
        freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=30_000.0, thermal_occupation=0.02
    )
    r = Resonator(freq=7.0, internal_quality_factor=5000.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08, label="cap0")])

    reduced = eliminate(chip, "r").chip
    jumps = [reduced.backend.to_array(op) for op in reduced.backend._collapse_operators(reduced.resolve(frame="lab"))]
    angle = 0.08 / -2.0
    expected_down = 1.02 * np.cos(angle)**2 / 30_000.0 + np.sin(angle)**2 * (2 * np.pi * 7.0 / 5000.0)
    assert sum(abs(op[0, 1])**2 for op in jumps) == pytest.approx(expected_down, rel=1e-9)
    assert sum(abs(op[1, 0])**2 for op in jumps) == pytest.approx(.02 * np.cos(angle)**2 / 30_000.0, rel=1e-9)


def test_eliminate_three_survivors_emits_pairwise_edges_matching_yan_formula():
    """A mode touching three survivors emits one edge per pair, each carrying the Yan mediated J."""
    from quchip.chip.transformations import eliminate

    qs = [DuffingTransmon(freq=5.0 + 0.15 * i, anharmonicity=-0.25, levels=3, label=f"q{i}") for i in range(3)]
    bus = Resonator(freq=7.0, levels=4, label="bus")
    chip = Chip(qs + [bus], couplings=[Capacitive(q, bus, g=0.06, label=f"leg{i}") for i, q in enumerate(qs)])

    res = eliminate(chip, "bus")
    reduced = res.chip
    assert sorted(d.label for d in reduced.devices) == ["q0", "q1", "q2"]
    pairs = {frozenset((c.device_a_label, c.device_b_label)) for c in reduced.couplings}
    assert pairs == {frozenset({"q0", "q1"}), frozenset({"q0", "q2"}), frozenset({"q1", "q2"})}
    assert all(type(c) is Capacitive for c in reduced.couplings)

    exchange = res.effective_params["exchange"]
    assert set(exchange) == {("q0", "q1"), ("q0", "q2"), ("q1", "q2")}
    freqs = {q.label: q.freq for q in qs}
    for (a, b), entry in exchange.items():
        delta_a = freqs[a] - 7.0
        delta_b = freqs[b] - 7.0
        expected = 0.06 * 0.06 / 2.0 * (1.0 / delta_a + 1.0 / delta_b)
        assert float(entry["j_eff"]) == pytest.approx(expected, rel=1e-6)


def test_eliminate_unknown_method_raises_value_error():
    """eliminate() rejects any method other than 'sw'/'exact'."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05)])
    with pytest.raises(ValueError, match="'sw'|'exact'"):
        eliminate(chip, "r", method="numeric")


@pytest.mark.optional_backend
def test_eliminate_bridge_exchange_is_differentiable_in_leg_g():
    """The mediated exchange J is differentiable in a bridge leg's coupling strength g."""
    pytest.importorskip("dynamiqs")
    import jax
    from quchip.chip.transformations import eliminate

    def j_eff(g):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
        q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="q1")
        bus = Resonator(freq=7.0, levels=3, label="bus")
        chip = Chip(
            [q0, q1, bus],
            couplings=[Capacitive(q0, bus, g=g), Capacitive(q1, bus, g=0.05)],
            backend="dynamiqs",
        )
        return eliminate(chip, "bus").effective_params["exchange"]["j_eff"]

    grad = jax.grad(j_eff)(0.05)
    # dJ/dg1 = (g2/2)(1/Δ1 + 1/Δ2)
    assert float(grad) == pytest.approx(0.05 / 2.0 * (1.0 / -2.0 + 1.0 / -1.9), rel=1e-6)


def test_eliminate_bridge_composes_sequentially_over_a_chain():
    """Eliminating both couplers of QB1-TC1-CR-TC2-QB2 leaves two mediated qubit-resonator couplings."""
    from quchip.chip.transformations import eliminate

    qb1 = DuffingTransmon(freq=4.7, anharmonicity=-0.2, levels=2, label="qb1")
    tc1 = DuffingTransmon(freq=6.0, anharmonicity=-0.1, levels=2, label="tc1")
    cr = Resonator(freq=4.2, levels=3, label="cr")
    tc2 = DuffingTransmon(freq=6.0, anharmonicity=-0.1, levels=2, label="tc2")
    qb2 = DuffingTransmon(freq=4.5, anharmonicity=-0.2, levels=2, label="qb2")
    chip = Chip(
        [qb1, tc1, cr, tc2, qb2],
        couplings=[
            Capacitive(qb1, tc1, g=0.09),
            Capacitive(tc1, cr, g=0.11),
            Capacitive(tc2, cr, g=0.11),
            Capacitive(qb2, tc2, g=0.10),
        ],
    )

    step1 = eliminate(chip, "tc1")
    step2 = eliminate(step1.chip, "tc2")
    reduced = step2.chip

    assert sorted(d.label for d in reduced.devices) == ["cr", "qb1", "qb2"]
    pairs = {frozenset((c.device_a_label, c.device_b_label)) for c in reduced.couplings}
    assert pairs == {frozenset({"qb1", "cr"}), frozenset({"qb2", "cr"})}


def test_eliminate_folds_through_a_fold_created_edge():
    """A fold-created coupling behaves as an ordinary edge later, driving the next survivor's g/Δ and Lamb shift."""
    from quchip.chip.transformations import eliminate

    step1 = eliminate(_bridge_chip(), "bus")
    mid = step1.chip
    (edge,) = mid.couplings
    assert type(edge) is Capacitive

    step2 = eliminate(mid, "q1")
    assert [d.label for d in step2.chip.devices] == ["q0"]

    delta = float(step1.effective_params["q0"]["freq_after"] - step1.effective_params["q1"]["freq_after"])
    g = float(edge.g)
    assert step2.validity["elim_bus"]["g_over_delta"] == pytest.approx(abs(g / delta), rel=1e-9)
    # Deep in the dispersive regime (g0/Δ ≈ 0.013) the leaf fold's Lamb shift
    # is g0²/Δ to leading order; the counter-rotating correction is ~1%.
    lamb = float(step2.effective_params["q0"]["lamb_shift"])
    assert lamb == pytest.approx(g**2 / delta, rel=0.05)
    # The exact route reads the same g and must agree on the dressed frequency.
    exact = eliminate(mid, "q1", method="exact")
    assert float(exact.chip.freq("q0")) == pytest.approx(float(step2.chip.freq("q0")), abs=1e-6)


def test_eliminate_retains_thermal_bath_and_intrinsic_loss():
    """Independent thermal channels and intrinsic loss retain their mapped generator."""
    from quchip import Bath, eliminate
    from qutip import lindblad_dissipator

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=7.0, internal_quality_factor=5000.0, levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.05)],
                baths=[Bath("thermal", targets=[q, r], temperature=20., rate=.01)])
    result = eliminate(chip, "r")
    before = chip.backend._collapse_operators(chip.resolve(frame="lab"))
    after = result.chip.backend._collapse_operators(result.chip.resolve(frame="lab"))
    expected = sum(lindblad_dissipator(result.mapping.project_operator(c)) for c in before)
    actual = sum(lindblad_dissipator(c) for c in after)
    np.testing.assert_allclose(actual.full(), expected.full(), atol=1e-12)


@pytest.mark.optional_backend
def test_eliminate_lamb_shift_is_differentiable_in_g():
    """The Lamb shift is differentiable in g on a JAX-native backend, since chi is lazy."""
    pytest.importorskip("dynamiqs")
    import jax
    from quchip.chip.transformations import eliminate

    def lamb_shift(g):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)], backend="dynamiqs")
        return eliminate(chip, "r").effective_params["q"]["lamb_shift"]

    grad = jax.grad(lamb_shift)(0.08)
    # d/dg (g^2/Δ) = 2g/Δ, Δ = -2.0
    assert float(grad) == pytest.approx(2 * 0.08 / (5.0 - 7.0), rel=1e-4)


def test_eliminate_with_circuit_level_survivor_retains_the_shift():
    """A survivor without a 'freq' tunable keeps its bare spectrum; the shift is reported."""
    from quchip import ChargeBasisTransmon
    from quchip.chip.transformations import eliminate

    q = ChargeBasisTransmon(E_C=0.25, E_J=12.0, n_g=0.0, levels=4, label="q")
    r = Resonator(freq=7.1, levels=5, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08)])
    bare_freq = chip["q"].freq

    res = eliminate(chip, r)

    assert [d.label for d in res.chip.devices] == ["q"]
    assert res.chip["q"].freq == pytest.approx(bare_freq)  # spectrum not folded
    delta = bare_freq - 7.1
    energy_vectors = np.asarray(q.eigenvectors())
    charge = energy_vectors.conj().T @ np.asarray(q.charge_coupling_operator()) @ energy_vectors
    lamb = (0.08 * abs(charge[0, 1])) ** 2 / delta
    assert float(res.effective_params["q"]["lamb_shift"]) == pytest.approx(lamb, rel=0.05)
    assert float(res.effective_params["q"]["freq_after"]) == pytest.approx(bare_freq + lamb, rel=1e-3)


def test_eliminate_coupling_target_reports_both_shifts_without_parameter_folding():
    """Both endpoint shifts are reported while their authored parameters stay fixed."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05, label="c")])

    res = eliminate(chip, "c")
    reduced = res.chip

    assert sorted(d.label for d in reduced.devices) == ["q", "r"]
    assert not reduced.couplings
    assert reduced["q"].freq == q.freq and reduced["r"].freq == r.freq

    assert res.validity["c"]["g_over_delta"] == pytest.approx(abs(0.05 / (5.0 - 7.0)), rel=1e-6)
    assert float(res.effective_params["q"]["lamb_shift"]) != 0.0
    assert float(res.effective_params["r"]["lamb_shift"]) != 0.0


def test_eliminate_coupling_target_tunable_capacitive_reports_validity():
    """Tunable-capacitive elimination reports validity metrics."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    chip = Chip([q, r], couplings=[TunableCapacitive(q, r, g_0=0.05, label="tc")])

    res = eliminate(chip, "tc")
    assert res.validity != {}
    assert res.validity["tc"]["g_over_delta"] == pytest.approx(abs(0.05 / (5.0 - 7.0)), rel=1e-6)


def test_eliminate_coupling_target_retains_a_diagonal_interaction():
    """Removing an already diagonal edge leaves its matrix and coordinates unchanged."""
    from quchip.chip.transformations import eliminate

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    chip = Chip([q, r], couplings=[CrossKerr(q, r, chi=0.001, label="xk")])

    result = eliminate(chip, "xk")
    np.testing.assert_allclose(result.chip.hamiltonian().matrix(), chip.hamiltonian().matrix(), atol=1e-12)
    np.testing.assert_allclose(result.mapping.embedding, np.eye(np.prod(chip.dims)), atol=1e-12)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_eliminate_coupling_target_keeps_circuit_parameters(method):
    """Circuit devices need no artificial frequency parameter to carry a correction."""
    from quchip import ChargeBasisTransmon
    from quchip.chip.transformations import eliminate

    q = ChargeBasisTransmon(E_C=0.25, E_J=12.0, n_g=0.0, levels=4, label="q")
    r = Resonator(freq=7.1, levels=5, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08, label="c")], basis="eigen")
    result = eliminate(chip, "c", method=method)
    assert result.chip["q"].E_C == q.E_C and result.chip["q"].E_J == q.E_J
    assert np.isfinite(result.effective_params["q"]["freq_after"])
    if method == "exact":
        from quchip import Exact
        source_h = chip.resolve(frame="lab", approximation=Exact()).hamiltonian().matrix()
        target_h = result.chip.resolve(frame="lab").hamiltonian().matrix()
        np.testing.assert_allclose(result.mapping.project_operator(source_h).full(), target_h, atol=1e-11)
