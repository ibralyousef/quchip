"""Physics contracts for isolated and dressed transition APIs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import RWA, Capacitive, Chip, DuffingTransmon, Exact, Resonator
from quchip.backend import get_default_backend
from quchip.declarative.expr import materialize_array
from quchip.devices import ChargeBasisTransmon


def test_isolated_transition_uses_energy_states_in_the_authored_basis() -> None:
    device = ChargeBasisTransmon(
        E_C=0.25,
        E_J=15.0,
        n_g=0.17,
        num_basis=9,
        levels=4,
        basis="eigen",
        label="q",
    )
    hamiltonian = np.asarray(materialize_array(device.unresolved_hamiltonian()))
    energies, vectors = np.linalg.eigh(hamiltonian)

    transition = np.asarray(get_default_backend().to_array(device.transition(0, 2)))
    in_energy_basis = vectors.conj().T @ transition @ vectors
    expected_magnitudes = np.zeros_like(in_energy_basis)
    expected_magnitudes[0, 2] = 1.0
    expected_magnitudes[2, 0] = 1.0

    np.testing.assert_allclose(np.abs(in_energy_basis), expected_magnitudes, atol=1e-11)
    np.testing.assert_allclose(transition, transition.conj().T, atol=1e-11)
    assert device.transition_frequency(0, 2) == pytest.approx(energies[2] - energies[0])
    with pytest.raises(ValueError, match="dimension 4"):
        device.transition_frequency(0, 4)


@pytest.mark.validation
def test_isolated_transition_frequency_is_jax_transformable() -> None:
    def gap(frequency):
        device = DuffingTransmon(
            freq=frequency,
            anharmonicity=-0.25,
            levels=3,
            label="q",
        )
        return device.transition_frequency(0, 2)

    frequencies = jnp.asarray([4.8, 5.0, 5.2])
    expected = 2.0 * frequencies - 0.25

    np.testing.assert_allclose(jax.jit(jax.vmap(gap))(frequencies), expected)
    assert jax.grad(gap)(5.0) == pytest.approx(2.0)


@pytest.fixture
def dressed_chip():
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    resonator = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([qubit, resonator], [Capacitive(qubit, resonator, g=0.05)])
    return chip, qubit, resonator


@pytest.mark.parametrize("approximation", [RWA(), Exact()])
def test_dressed_transition_follows_chip_approximation(approximation) -> None:
    """A dressed transition uses the same retained Hamiltonian as a lab-frame solve."""
    first = Resonator(freq=5.0, levels=3, label="first")
    second = Resonator(freq=7.0, levels=3, label="second")
    chip = Chip([first, second], [Capacitive(first, second, g=0.2)], approximation=approximation)
    energies = chip.resolve(frame="lab").dress().dressed_eigenvalues
    assert chip.transition_frequency(first, 0, 1) == pytest.approx(energies[(1, 0)] - energies[(0, 0)], abs=1e-12)


def test_dressed_transition_rejects_target_in_condition(dressed_chip) -> None:
    chip, qubit, _ = dressed_chip

    with pytest.raises(ValueError, match="target"):
        chip.transition_frequency(qubit, 0, 1, when={qubit: 2})


def test_dressed_transition_rejects_non_mapping_condition(dressed_chip) -> None:
    chip, qubit, _ = dressed_chip

    with pytest.raises(TypeError, match="mapping"):
        chip.transition_frequency(qubit, 0, 1, when=[("r", 1)])  # type: ignore[arg-type]


def test_dressed_transition_rejects_ambiguous_assignment() -> None:
    first = Resonator(freq=6.0, levels=3, label="first")
    second = Resonator(freq=6.0, levels=3, label="second")
    chip = Chip([first, second], [Capacitive(first, second, g=0.05)])

    with pytest.raises(ValueError, match="assignment is unreliable"):
        chip.transition_frequency(first, 0, 1)


@pytest.mark.validation
def test_dressed_transition_frequency_is_jax_transformable() -> None:
    pytest.importorskip("dynamiqs")

    def gap(frequency):
        qubit = DuffingTransmon(
            freq=frequency,
            anharmonicity=-0.25,
            levels=3,
            label="q",
        )
        chip = Chip([qubit], backend="dynamiqs")
        return chip.transition_frequency(qubit, 0, 2)

    frequencies = jnp.asarray([4.8, 5.0, 5.2])
    expected = 2.0 * frequencies - 0.25

    np.testing.assert_allclose(jax.jit(jax.vmap(gap))(frequencies), expected)
    assert jax.grad(gap)(5.0) == pytest.approx(2.0)
