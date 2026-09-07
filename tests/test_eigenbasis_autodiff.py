"""Forward- and reverse-mode derivatives through the local eigensolver."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from quchip import Chip, Resonator  # noqa: E402
from quchip.chip.couplings import Capacitive  # noqa: E402
from quchip.devices.transmon.duffing import DuffingTransmon  # noqa: E402
from quchip.engine.basis import _differentiable_eigenpairs  # noqa: E402


def _dispersive_chip() -> Chip:
    pytest.importorskip("dynamiqs")
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    resonator = Resonator(freq=7.0, levels=4, label="r")
    return Chip(
        [qubit, resonator], couplings=[Capacitive(qubit, resonator, g=0.1)], frame="rotating", backend="dynamiqs"
    )


def _resonator_frame(chip: Chip, freq: float) -> float:
    rebound = chip.with_params({"q.freq": freq})
    return rebound.resolve().resolved_frame.frequencies["r"]


def test_forward_and_reverse_derivatives_agree_with_finite_differences() -> None:
    """jacfwd and grad through a traced device frequency both match a central difference."""
    chip = _dispersive_chip()
    step = 1e-4
    reference = (_resonator_frame(chip, 5.0 + step) - _resonator_frame(chip, 5.0 - step)) / (2 * step)
    forward = float(jax.jacfwd(lambda f: _resonator_frame(chip, f))(5.0))
    reverse = float(jax.grad(lambda f: _resonator_frame(chip, f))(5.0))
    assert forward == pytest.approx(reference, rel=1e-4)
    assert reverse == pytest.approx(reference, rel=1e-4)
    assert abs(reference) > 1e-4


def test_hessian_is_finite_away_from_degeneracies() -> None:
    """Second derivatives are finite and match a second difference at a nondegenerate operating point."""
    chip = _dispersive_chip()
    curvature = float(jax.hessian(lambda f: _resonator_frame(chip, f))(5.0))
    step = 1e-3
    reference = (
        _resonator_frame(chip, 5.0 + step) - 2 * _resonator_frame(chip, 5.0) + _resonator_frame(chip, 5.0 - step)
    ) / step**2
    assert np.isfinite(curvature)
    assert curvature == pytest.approx(reference, rel=5e-2)


def test_eigenvector_tangent_is_orthogonal_to_the_eigenvector() -> None:
    """The parallel-transport gauge keeps v_i^dagger dv_i = 0 for every retained eigenvector."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    matrix = jnp.asarray(base + base.conj().T)
    direction = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    direction = jnp.asarray(direction + direction.conj().T)
    (_, vectors), (_, dvectors) = jax.jvp(lambda m: _differentiable_eigenpairs(m, 3), (matrix,), (direction,))
    connection = jnp.diagonal(jnp.conj(vectors).T @ dvectors)
    np.testing.assert_allclose(np.asarray(connection), 0.0, atol=1e-10)


@pytest.mark.parametrize("origin", [-1e8, 1e8])
def test_eigenprojector_derivative_is_independent_of_energy_origin(origin: float) -> None:
    """An identity energy shift cannot suppress a nondegenerate physical response."""
    matrix = jnp.array([[0.0, 0.02], [0.02, 0.04]])
    direction = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    def projector(m):
        _, vectors = _differentiable_eigenpairs(m, 1)
        return vectors @ vectors.conj().T

    step = 1e-6
    finite_difference = (projector(matrix + step * direction) - projector(matrix - step * direction)) / (2 * step)
    shifted = matrix + origin * jnp.eye(2)
    _, tangent = jax.jvp(projector, (shifted,), (direction,))
    np.testing.assert_allclose(tangent, finite_difference, rtol=1e-6, atol=1e-6)


def test_canonical_energy_vector_phase_and_derivative_agree():
    """The largest authored component is positive and its local phase derivative is correct."""
    from quchip.engine.basis import resolve_local_basis

    def vectors(value):
        matrix = jnp.array([[0.0, value - 0.2j], [value + 0.2j, 1.0]])
        return resolve_local_basis(matrix).energy_vectors

    value = 0.3
    actual = vectors(value)
    pivots = actual[jnp.argmax(jnp.abs(actual), axis=0), jnp.arange(2)]
    np.testing.assert_allclose(pivots.imag, 0.0, atol=1e-12)
    assert np.all(np.asarray(pivots.real) > 0)
    step = 1e-5
    finite_difference = (vectors(value + step) - vectors(value - step)) / (2 * step)
    np.testing.assert_allclose(jax.jacfwd(vectors)(value), finite_difference, atol=1e-8)
    def loss(x):
        return jnp.real(vectors(x)[0, 1])
    reference = (loss(value + step) - loss(value - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(value), reference, atol=1e-8)
