"""Tests for KerrCavity Hamiltonian eigenvalues, TwoPhotonDrive channels, and cat-state prep."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.devices.kerr_cavity import KerrCavity
from quchip.control.drives_two_photon import TwoPhotonDrive


class TestKerrCavityHamiltonian:
    """Verify KerrCavity eigenvalues against the analytical formula."""

    def test_eigenvalues_analytical(self):
        """E_n = omega*n - K*n*(n-1)."""
        omega = 5.0
        K = 1.0
        levels = 8
        cav = KerrCavity(freq=omega, kerr=K, levels=levels, label="cav")
        from quchip.backend import get_default_backend
        backend = get_default_backend()
        evals = np.sort(np.linalg.eigvalsh(cav.hamiltonian().matrix(backend=backend)).real)
        expected = np.array([omega * n - K * n * (n - 1) for n in range(levels)])
        expected_sorted = np.sort(expected)
        npt.assert_allclose(evals, expected_sorted, atol=1e-10)

    def test_zero_kerr_is_harmonic(self):
        """K=0 should give equally spaced harmonic oscillator levels."""
        omega = 3.0
        cav = KerrCavity(freq=omega, kerr=0.0, levels=6, label="cav")
        from quchip.backend import get_default_backend
        backend = get_default_backend()
        evals = np.sort(np.linalg.eigvalsh(cav.hamiltonian().matrix(backend=backend)).real)
        expected = np.array([omega * n for n in range(6)])
        npt.assert_allclose(evals, expected, atol=1e-10)

    def test_hamiltonian_is_hermitian(self):
        """H must be Hermitian."""
        cav = KerrCavity(freq=4.0, kerr=0.5, levels=10, label="cav")
        H_arr = np.asarray(cav.hamiltonian().matrix())
        npt.assert_allclose(H_arr, H_arr.conj().T, atol=1e-12)

    def test_negative_kerr_raises(self):
        """Negative kerr should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            KerrCavity(freq=5.0, kerr=-0.1, levels=5, label="bad")

    def test_zero_freq_raises(self):
        """Zero or negative freq should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            KerrCavity(freq=0.0, kerr=1.0, levels=5, label="bad")

    def test_auto_label(self):
        """Auto-label should use 'kerr_cavity_' prefix."""
        from quchip.utils.labeling import reset_label_counters
        reset_label_counters()
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=5)
        assert cav.label.startswith("kerr_cavity_")

    def test_computational_property(self):
        """A Kerr cavity is not automatically treated as a computational qubit."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=5, label="cav")
        assert cav.computational is False

class TestTwoPhotonDrive:
    """Verify the two-photon operator and carrier scheduling."""

    def test_two_photon_operator_matches_oscillator_matrix(self):
        """The full operator equals a² + a†², including amplitudes and support."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="cav")
        d2 = TwoPhotonDrive(target=cav)
        from quchip.control.signal import AnalyticSignal
        from quchip.engine.ir import Constant

        op_arr = np.asarray(d2.hamiltonian(cav, AnalyticSignal(Constant(1.0))).matrix(t=0.0))
        lowering = np.diag(np.sqrt(np.arange(1, cav.levels)), k=1)
        npt.assert_allclose(op_arr, lowering @ lowering + lowering.T @ lowering.T, atol=1e-12)

    def test_auto_label(self):
        """Auto-label should use 'two_photon_' prefix."""
        from quchip.utils.labeling import reset_label_counters
        reset_label_counters()
        d2 = TwoPhotonDrive()
        assert d2.label.startswith("two_photon_")

    def test_schedule_without_freq_builds_a_carrier_free_signal(self):
        from quchip import Chip, QuantumSequence
        from quchip.control.envelopes import Square

        cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="cav")
        d2 = TwoPhotonDrive(target=cav)
        chip = Chip([cav])
        chip.wire(d2)
        seq = QuantumSequence(chip)
        seq.schedule(d2, envelope=Square(duration=20.0, amplitude=0.1))
        assert seq.scheduled_ops[0].freq is None


class TestCatStatePreparation:
    """Physics integration test: rotating-frame adiabatic ramp -> cat state."""

    def test_mean_photon_number_at_alpha_squared(self):
        """After adiabatic ramp, <n> should approach eps2/K = alpha^2."""
        from quchip import Chip, QuantumSequence
        from quchip.control.envelopes import LinearRamp

        K = 0.01       # GHz; the retained Fock ladder stays energy ordered
        omega = 5.0    # GHz
        eps2_max = 2.0 * K    # alpha^2 = 2.0
        N_fock = 20
        t_ramp = 40.0 / K  # ns; fixed adiabaticity relative to Kerr splitting
        T_total = 45.0 / K
        n_steps = 450

        cav = KerrCavity(freq=omega, kerr=K, levels=N_fock, label="cav")
        d2 = TwoPhotonDrive(target=cav)

        chip = Chip([cav], frame="rotating")
        chip.wire(d2)

        n_op = cav.number_operator()
        e_ops = {cav: [n_op]}

        seq = QuantumSequence(chip)
        # In rotating frame, drive at 2*omega; amplitude doubled for RWA factor
        seq.schedule(d2, envelope=LinearRamp(T_total, t_ramp, amplitude=2 * eps2_max),
                     freq=2 * omega)

        tlist = np.linspace(0, T_total, n_steps)
        # Prepare the Fock vacuum; the truncated negative-Kerr Hamiltonian's
        # lowest energy lies at the cutoff and is not the cat-preparation state.
        result = seq.simulate(tlist=tlist, initial_state={cav: cav.basis_state(0)},
                              e_ops=e_ops, options={"nsteps": 5000})

        n_final = float(np.real(result.expect_final(cav, index=0)))
        assert abs(n_final - eps2_max / K) < 0.5, (
            f"<n> = {n_final:.3f} but expected approx {eps2_max/K:.1f}"
        )
