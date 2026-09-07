"""Regression coverage for QuTiPBackend.resolve_solver_options' max_step consumption."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.chip.chip import Chip
from quchip.devices.transmon.duffing import DuffingTransmon


class TestMaxStepGuardPolicy:
    """resolve_solver_options' max_step insertion/authority policy, tested directly on the dict contract."""

    def _backend(self):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        return Chip(devices=[q], backend="qutip").backend

    def _resolve(self, options: dict, max_step_ns) -> dict:
        return self._backend().resolve_solver_options(
            options, metadata={"max_step_ns": max_step_ns}, tlist=np.array([0.0, 308.0])
        )

    @pytest.mark.parametrize("max_step_ns", [0.5], ids=["sub-ns"])
    def test_concrete_positive_finite_metadata_is_inserted(self, max_step_ns: float) -> None:
        """A concrete, positive, finite max_step_ns hint is inserted as max_step when unset."""
        resolved = self._resolve({}, max_step_ns)
        assert resolved["max_step"] == max_step_ns

    @pytest.mark.parametrize(
        "max_step_ns", [0.0, -1.0, float("nan"), float("inf")], ids=["zero", "negative", "nan", "inf"]
    )
    def test_non_positive_or_non_finite_metadata_is_ignored(self, max_step_ns: float) -> None:
        """Zero, negative, NaN, and infinite max_step_ns hints never insert a max_step."""
        resolved = self._resolve({}, max_step_ns)
        assert "max_step" not in resolved

    @pytest.mark.parametrize("user_max_step", [1.0, 2.5, 0], ids=["typical", "fractional", "zero-unbounded"])
    def test_explicit_user_max_step_is_always_authoritative(self, user_max_step) -> None:
        """An explicit user max_step -- including QuTiP's own unbounded 0 -- is never overridden by the hint."""
        resolved = self._backend().resolve_solver_options(
            {"max_step": user_max_step}, metadata={"max_step_ns": 4.0}, tlist=np.array([0.0, 308.0])
        )
        assert resolved["max_step"] == user_max_step


def test_user_step_budget_is_preserved_with_automatic_pulse_cap():
    from quchip.backend.qutip import QuTiPBackend

    options = QuTiPBackend().resolve_solver_options(
        {"nsteps": 1234}, metadata={"max_step_ns": 0.01}, tlist=np.array([0.0, 308.0])
    )
    assert options["nsteps"] == 1234
    assert options["max_step"] == 0.01
