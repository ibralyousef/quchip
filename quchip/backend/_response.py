"""Dense response algebra shared by NumPy and JAX backends."""

from typing import Any

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
        residuals=residuals,
        _condition_numbers=DeferredValue(lambda: xp.linalg.cond(systems())),
    )
