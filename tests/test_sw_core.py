"""Schrieffer-Wolff kernels reproduce closed-form second-order results."""

from __future__ import annotations

from quchip.approximations import Exact, RWA

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator
from quchip.chip.sw import (
    bare_hamiltonian,
    h_effective_second_order,
    mode_blocks,
    extract_pair_parameters,
    pathway_attribution,
    sylvester_generator,
)


def _two_qubit_exchange(omega_a: float, omega_b: float, g: float) -> jnp.ndarray:
    """H for two two-level systems with an RWA exchange, C-order product basis (na, nb)."""
    h = jnp.diag(jnp.array([0.0, omega_b, omega_a, omega_a + omega_b], dtype=complex))
    h = h.at[1, 2].set(g).at[2, 1].set(g)  # |01><10| + h.c.
    return h


def test_two_level_exchange_reproduces_level_repulsion():
    """Eliminating either qubit shifts its partner's frequency by g²/Δ, sign flipped by which mode is eliminated."""
    omega_a, omega_b, g = 5.0, 5.3, 0.02
    delta = omega_a - omega_b
    h = _two_qubit_exchange(omega_a, omega_b, g)
    dims, labels = (2, 2), ["a", "b"]

    p_mask, _ = mode_blocks(dims, labels, "b")
    s, _ = sylvester_generator(h, p_mask)
    h_eff = h_effective_second_order(h, s, p_mask)
    params = extract_pair_parameters(h_eff, np.flatnonzero(p_mask), labels, dims, "b")
    assert abs(float(params["a"]["freq_after"]) - (omega_a + g**2 / delta)) < 1e-12

    # Repulsion is symmetric: eliminating the other mode pushes the other way.
    p_mask_a, _ = mode_blocks(dims, labels, "a")
    s_a, _ = sylvester_generator(h, p_mask_a)
    h_eff_a = h_effective_second_order(h, s_a, p_mask_a)
    params_a = extract_pair_parameters(h_eff_a, np.flatnonzero(p_mask_a), labels, dims, "a")
    assert abs(float(params_a["b"]["freq_after"]) - (omega_b - g**2 / delta)) < 1e-12


def test_bare_hamiltonian_uses_the_selected_approximation():
    """SW input retains exchange under RWA and counter-rotating terms under Exact."""
    def build(approximation):
        q = Resonator(freq=5.0, levels=3, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        return Chip([q, r], [Capacitive(q, r, g=0.05)], approximation=approximation)

    h_rwa, _, _ = bare_hamiltonian(build(RWA()))
    h_exact, _, _ = bare_hamiltonian(build(Exact()))

    assert h_rwa[3, 1] == 0.05
    assert h_exact[3, 1] == 0.05
    assert h_rwa[0, 4] == 0.0
    assert h_exact[0, 4] == 0.05


def _bridge_h():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=6.3, levels=4, label="bus")
    chip = Chip(
        [q0, q1, bus],
        couplings=[Capacitive(q0, bus, g=0.08, label="leg0"), Capacitive(q1, bus, g=0.08, label="leg1")],
        approximation=RWA(),
    )
    return bare_hamiltonian(chip)


def test_degenerate_cross_block_with_zero_coupling_is_guarded():
    """Exactly degenerate P/Q levels with no matrix element between them: no NaN, finite grad."""
    omega = 5.0
    dims, labels = (2, 2), ["a", "b"]
    p_mask, _ = mode_blocks(dims, labels, "b")

    def ground_energy(g):
        h = jnp.diag(jnp.array([0.0, omega, omega, 2.0 * omega], dtype=complex))
        h = h.at[0, 3].set(g).at[3, 0].set(g)  # couples |00> <-> |11| only
        s, _ = sylvester_generator(h, p_mask)
        assert bool(jnp.all(jnp.isfinite(s)))
        h_eff = h_effective_second_order(h, s, p_mask)
        return h_eff[0, 0].real

    grad = jax.grad(ground_energy)(0.05)
    assert np.isfinite(float(grad))
    # d/dg of the 2nd-order shift g^2/(0 - 2*omega) = -g/omega.
    assert abs(float(grad) - (-0.05 / omega)) < 1e-12


def test_pathway_attribution_sums_to_commutator_element():
    """Summed pathway amounts equal the commutator element 0.5*(S@V-V@S); only the bus pathway contributes."""
    h, labels, dims = _bridge_h()
    p_mask, _ = mode_blocks(dims, labels, "bus")
    s, _ = sylvester_generator(h, p_mask)

    p_index = np.flatnonzero(p_mask)
    occ = np.array(np.unravel_index(np.arange(int(np.prod(dims))), dims))
    i_idx = int(np.flatnonzero((occ[0] == 1) & (occ[1] == 0) & (occ[2] == 0))[0])
    j_idx = int(np.flatnonzero((occ[0] == 0) & (occ[1] == 1) & (occ[2] == 0))[0])
    assert i_idx in p_index and j_idx in p_index

    paths = pathway_attribution(h, s, p_mask, i_idx, j_idx)
    total = sum(amount for _, amount in paths)
    e_diag = jnp.diag(jnp.diagonal(h))
    v = h - e_diag
    expected = (0.5 * (s @ v - v @ s))[i_idx, j_idx]
    assert abs(complex(total) - complex(expected)) < 1e-12
    # The bus single-excitation state is the only virtual pathway here.
    bus_idx = int(np.flatnonzero((occ[0] == 0) & (occ[1] == 0) & (occ[2] == 1))[0])
    assert [k for k, _ in paths] == [bus_idx]


@pytest.mark.parametrize("coupling", [0.02, 1e-15])
def test_coupled_degenerate_cross_block_is_rejected(coupling):
    """Every nonzero resonant P/Q coupling makes this perturbative generator singular."""
    mask, _ = mode_blocks((2, 2), ["a", "b"], "b")
    with pytest.raises(ValueError, match="degenerate.*coupl"):
        sylvester_generator(_two_qubit_exchange(5.0, 5.0, coupling), mask)


def test_coupled_degenerate_cross_block_fails_under_jit_and_vmap():
    """Tracing cannot turn singular reduction into a plausible finite answer."""
    mask, _ = mode_blocks((2, 2), ["a", "b"], "b")

    def reduced_frequency(freq):
        h = _two_qubit_exchange(freq, 5.0, 0.02)
        s, _ = sylvester_generator(h, mask)
        return h_effective_second_order(h, s, mask)[1, 1].real

    for calculate, argument in (
        (jax.jit(reduced_frequency), 5.0),
        (jax.jit(jax.vmap(reduced_frequency)), jnp.array([4.8, 5.0])),
    ):
        with pytest.raises(Exception, match="degenerate.*coupl"):
            jax.block_until_ready(calculate(argument))


def test_valid_sw_jit_batched_values_and_derivatives():
    """Valid reductions retain closed-form derivatives through JIT and batching."""
    mask, _ = mode_blocks((2, 2), ["a", "b"], "b")

    def reduced_frequency(freq):
        h = _two_qubit_exchange(freq, 5.0, 0.02)
        s, _ = sylvester_generator(h, mask)
        return h_effective_second_order(h, s, mask)[1, 1].real

    frequencies = jnp.array([4.8, 5.3])
    values, derivatives = jax.jit(jax.vmap(jax.value_and_grad(reduced_frequency)))(frequencies)
    np.testing.assert_allclose(values, frequencies + 0.02**2 / (frequencies - 5.0), atol=1e-12)
    np.testing.assert_allclose(derivatives, 1.0 - 0.02**2 / (frequencies - 5.0)**2, atol=1e-12)
