"""Pure-dephasing rates, coherence decay and preserved populations."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quchip.backend.protocol import Backend
from quchip.chip.chip import Chip
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate


class TestSolverRouting:
    """Verify that simulate selects the correct solver branch."""

    def test_T2_without_T1_emits_pure_dephasing(self) -> None:
        """T2 without T1 emits a single pure-dephasing operator with gamma_phi = 1/T2."""
        T2 = 100.0
        q = DuffingTransmon(
            freq=5.0,
            anharmonicity=-0.25,
            levels=3,
            label="q",
            T2=T2,
        )
        c_ops = q.collapse_operators()
        assert len(c_ops) == 1, (
            f"Expected 1 collapse op (pure dephasing) when T2 is set and T1 is None, got {len(c_ops)}."
        )
        # Dephasing op is sqrt(2*gamma_phi) * n_hat with gamma_phi = 1/T2 here
        # (T1 absent) — n_hat[1,1] = 1, so op[1,1]^2 == 2/T2. The factor 2 makes
        # the 0–1 coherence decay at gamma_phi = 1/T2 (the input coherence time).
        mat = np.asarray(c_ops[0].full())
        coeff_sq = abs(mat[1, 1]) ** 2
        expected = 2.0 / T2
        assert abs(coeff_sq - expected) < 1e-12, (
            f"Dephasing coefficient squared {coeff_sq} should equal 2/T2 = {expected}."
        )

class TestDephasing:
    """Verify pure dephasing: off-diagonal density-matrix decay at rate 1/T2."""
    # L = sqrt(2*gamma_phi) * n_hat gives |rho_01(t)| = |rho_01(0)| * exp(-t/T2), with
    # gamma_phi = 1/T2 - 1/(2*T1); the factor 2 in L makes the input T2 equal the
    # resulting coherence time.

    T1 = 50_000  # ns  (large to isolate dephasing)
    T2 = 10_000  # ns
    FREQ = 5.0
    ALPHA = -0.25
    LEVELS = 3
    DURATION = 20_000  # 2 × T2
    N_POINTS = 51

    def test_offdiag_decay_rate(self, backend: Backend) -> None:
        """Off-diagonal |rho_01| decays at analytically predicted rate."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=self.T1,
            T2=self.T2,
        )
        chip = Chip([q])
        chip.set_frame("rotating")

        # Superposition |+⟩ = (|0⟩ + |1⟩) / √2
        ket0 = backend.basis(self.LEVELS, 0)
        ket1 = backend.basis(self.LEVELS, 1)
        psi_plus = (ket0 + ket1).unit()

        tlist = np.linspace(0, self.DURATION, self.N_POINTS)
        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi_plus,
            options={"nsteps": 5000},
        )

        offdiag = np.array([abs(complex(result.dm_at(t)[0, 1])) for t in tlist])

        # Analytical decay rate for n̂-based dephasing with the sqrt(2*gamma_phi)
        # normalization: rate = gamma_phi + 1/(2*T1) = 1/T2 (input coherence time).
        gamma_phi = 1.0 / self.T2 - 1.0 / (2.0 * self.T1)
        decay_rate = gamma_phi + 1.0 / (2.0 * self.T1)
        offdiag_analytic = 0.5 * np.exp(-decay_rate * tlist)

        npt.assert_allclose(
            offdiag,
            offdiag_analytic,
            atol=0.02,
            err_msg=(
                f"Off-diagonal decay deviates from analytic. "
                f"gamma_phi={gamma_phi:.2e}, decay_rate={decay_rate:.2e}. "
                f"Max |error| = {np.max(np.abs(offdiag - offdiag_analytic)):.4f}."
            ),
        )

    def test_populations_preserved_under_dephasing(self, backend: Backend) -> None:
        """Pure dephasing does not change diagonal populations: P0, P1 stay near 0.5 from |+⟩."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ALPHA,
            levels=self.LEVELS,
            label="q",
            T1=self.T1,
            T2=self.T2,
        )
        chip = Chip([q])
        chip.set_frame("rotating")

        ket0 = backend.basis(self.LEVELS, 0)
        ket1 = backend.basis(self.LEVELS, 1)
        psi_plus = (ket0 + ket1).unit()

        tlist = np.linspace(0, self.DURATION, self.N_POINTS)
        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi_plus,
            options={"nsteps": 5000},
        )

        p0 = result.population("q", 0)
        p1 = result.population("q", 1)

        # T1 is large relative to duration, so populations stay near 0.5
        # Allow 10% deviation for T1-induced drift
        assert np.all(p0 > 0.35), (
            f"P0 dropped below 0.35 — dephasing should not cause population transfer. Min P0 = {np.min(p0):.4f}."
        )
        assert np.all(p1 > 0.25), (
            f"P1 dropped below 0.25 — dephasing should not cause significant "
            f"population transfer. Min P1 = {np.min(p1):.4f}."
        )
