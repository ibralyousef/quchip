"""Tests for SimulationResult states-accessor guards."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from quchip.backend import get_default_backend, SolverResult
from quchip.results.results import _NO_STATES_MSG, SimulationResult


def _states_less_result(dev_dim: int = 2) -> SimulationResult:
    """A SimulationResult built with ``store_states`` off (``states=None``)."""
    backend = get_default_backend()
    solver_result = SolverResult(
        times=np.array([0.0, 1.0]),
        states=None,
        final_state=None,
        stats={},
        solver="sesolve",
    )
    return SimulationResult(
        solver_result=solver_result,
        backend=backend,
        dims=[dev_dim],
        device_info=[("q0", True)],
    )


def test_amplitude_array_without_states_raises_runtime_error() -> None:
    """amplitude_array() on a states-less result raises RuntimeError(_NO_STATES_MSG)."""
    result = _states_less_result()
    target = qt.basis(2, 0)
    with pytest.raises(RuntimeError, match=_NO_STATES_MSG):
        result.amplitude_array(target)


def test_list_history_does_not_cache_a_tracer():
    import jax
    from quchip.backend.dynamiqs import DynamiqsBackend
    from quchip.utils.jax_utils import contains_tracer

    backend = DynamiqsBackend()
    result = SimulationResult(
        SolverResult(times=np.array([0.0, 1.0]), states=[backend.basis(3, 0), backend.basis(3, 1)]),
        backend, [3], device_info=[("q", True)],
    )
    np.testing.assert_allclose(jax.jit(lambda: result.populations[(1,)])(), [0.0, 1.0])
    assert not contains_tracer(result._stacked_cache)
    assert isinstance(result.populations[(1,)], jax.Array)
    np.testing.assert_allclose(result.population("q", 1), [0.0, 1.0])
