"""Dense response algebra shared by NumPy and JAX backends."""

from typing import Any

import numpy as np

from quchip.backend.containers import LinearResponseSolverResult
from quchip.utils.values import DeferredValue


def stationary_condition_number(liouvillian: Any, dimension: int, *, xp: Any) -> Any:
    """Measure conditioning after replacing one equation with unit trace."""
    trace_row = xp.eye(dimension, dtype=liouvillian.dtype).reshape(1, -1)
    return xp.linalg.cond(xp.concatenate((liouvillian[:-1], trace_row), axis=0))


def linear_response(problem: Any, *, xp: Any) -> LinearResponseSolverResult:
    """Solve passive-linear scattering and capture deferred conditioning."""
    hamiltonian = xp.asarray(problem.hamiltonian, dtype=complex)
    couplings = xp.asarray(problem.couplings, dtype=complex)
    scattering = xp.asarray(problem.scattering, dtype=complex)
    frequencies = xp.atleast_1d(xp.array(problem.frequencies, dtype=float, copy=True))
    drift = -1j * hamiltonian - 0.5 * couplings.conj().T @ couplings
    indices = xp.asarray(problem.plane_indices)
    sources = -couplings.conj().T @ scattering[:, indices]

    def systems() -> Any:
        return -1j * 2.0 * xp.pi * frequencies[:, None, None] * xp.eye(drift.shape[0])[None, :, :] - drift[None, :, :]

    matrices = systems()
    amplitudes = xp.linalg.solve(matrices, sources[None, :, :])
    responses = scattering[xp.ix_(indices, indices)][None, :, :] + couplings[indices] @ amplitudes
    residuals = xp.linalg.norm(matrices @ amplitudes - sources[None, :, :], axis=(-2, -1))
    return LinearResponseSolverResult(
        responses=responses,
        mode_amplitudes=amplitudes,
        residuals=residuals,
        _condition_numbers=DeferredValue(lambda: xp.linalg.cond(systems())),
        _mode_covariance=DeferredValue(lambda: _mode_covariance(problem, drift, couplings, scattering, xp)),
    )


def _mode_covariance(problem: Any, drift: Any, couplings: Any, scattering: Any, xp: Any) -> Any:
    """Solve the passive Markov Lyapunov equation A N + N A† + D = 0."""
    occupations = xp.asarray([0.0 if c.input_occupation is None else c.input_occupation
                              for c in problem.field_channels]
                             + [0.0] * (scattering.shape[0] - len(problem.field_channels)))
    sources = -couplings.conj().T @ scattering
    diffusion = (sources * occupations) @ sources.conj().T
    # A common rotating frame cancels from the covariance equation.
    count = drift.shape[0]
    drift = drift - 1j*xp.imag(xp.trace(drift))/count*xp.eye(count)
    if xp is np:
        from scipy.linalg import solve_continuous_lyapunov

        covariance = solve_continuous_lyapunov(drift, -diffusion)
    else:
        identity = xp.eye(count)
        generator = xp.kron(drift, identity) + xp.kron(identity, drift.conj())
        covariance = xp.linalg.solve(generator, -diffusion.reshape(-1)).reshape((count, count))
    return (covariance + covariance.conj().T)/2
