"""Tests for FluxTunableTransmon, GaussianEdge, and QuantumSequence.flux_to.

Core tests — no backend simulation required for most; the flux_to test uses
the QuTiP backend for a minimal end-to-end check.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest


# ===========================================================================
# FluxTunableTransmon
# ===========================================================================


class TestFluxTunableTransmon:

    def test_hamiltonian_eigenvalues(self):
        """Duffing H eigenvalues: ground = 0, first ≈ freq, second ≈ 2*freq + alpha."""
        from quchip import FluxTunableTransmon

        freq = 4.47
        alpha = -0.2006
        q = FluxTunableTransmon(freq=freq, anharmonicity=alpha, levels=3)
        H = q.hamiltonian()
        evals = sorted(np.linalg.eigvalsh(H.matrix()).real)
        npt.assert_allclose(evals[0], 0.0, atol=1e-10)
        npt.assert_allclose(evals[1], freq, rtol=1e-8)
        npt.assert_allclose(evals[2], 2 * freq + alpha, rtol=1e-6)


    def test_frequency_at_nonzero_bias(self):
        """frequency_at(flux_bias) matches freq when constructed at that bias."""
        from quchip import FluxTunableTransmon

        freq = 4.0
        q = FluxTunableTransmon(freq=freq, anharmonicity=-0.20, flux_bias=0.1)
        npt.assert_allclose(float(q.frequency_at(0.1)), freq, rtol=1e-6)


    def test_flux_for_frequency_at_bias_matches_construction(self):
        """flux_for_frequency(q.freq) should return the original flux_bias."""
        from quchip import FluxTunableTransmon

        flux_bias = 0.15
        q = FluxTunableTransmon(freq=4.2, anharmonicity=-0.20, flux_bias=flux_bias)
        phi = float(q.flux_for_frequency(q.freq))
        npt.assert_allclose(phi, flux_bias, atol=1e-5)

    def test_asymmetric_squid(self):
        """Non-zero asymmetry should affect frequency at nonzero flux."""
        from quchip import FluxTunableTransmon

        q_sym = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, asymmetry=0.0)
        q_asym = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, asymmetry=0.1)
        npt.assert_allclose(float(q_sym.frequency_at(0.0)), float(q_asym.frequency_at(0.0)), rtol=1e-4)
        f_sym = float(q_sym.frequency_at(0.3))
        f_asym = float(q_asym.frequency_at(0.3))
        assert f_asym > f_sym, "Asymmetric SQUID should have higher freq at non-zero flux"

    def test_jax_traceable_construction(self):
        """jit over FluxTunableTransmon construction and frequency_at must not fail."""
        import jax

        from quchip import FluxTunableTransmon

        @jax.jit
        def get_freq_at(f, alpha, phi):
            q = FluxTunableTransmon(freq=f, anharmonicity=alpha)
            return q.frequency_at(phi)

        result = get_freq_at(4.47, -0.2006, 0.0)
        npt.assert_allclose(float(result), 4.47, rtol=1e-6)

    def test_jax_traceable_freq_sweep(self):
        """Sweeping freq over jnp.linspace — frequency_at stays traceable."""
        import jax
        import jax.numpy as jnp

        from quchip import FluxTunableTransmon

        @jax.jit
        def sweep(freqs):
            return jnp.array([
                FluxTunableTransmon(freq=f, anharmonicity=-0.20).frequency_at(0.1)
                for f in freqs
            ])

        freqs = jnp.linspace(4.0, 5.0, 5)
        results = sweep(freqs)
        assert results.shape == (5,)
        # Flux reduces frequency.
        assert jnp.all(results < freqs)


    def test_flux_bias_mutation_retunes_frequency_and_hamiltonian(self):
        """Changing the operating bias preserves the SQUID calibration and retunes the local model."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0, levels=3)
        expected_frequency = float(q.frequency_at(0.3))
        E_J_max_before = float(q._E_J_max)
        H_before = np.asarray(q.hamiltonian().matrix())

        q.flux_bias = 0.3

        H_after = np.asarray(q.hamiltonian().matrix())
        npt.assert_allclose(q.freq, expected_frequency, rtol=1e-10)
        npt.assert_allclose(float(q._E_J_max), E_J_max_before, rtol=1e-10)
        assert not np.allclose(H_before, H_after)
        npt.assert_allclose(np.linalg.eigvalsh(H_after)[1], expected_frequency, rtol=1e-10)


    def test_flux_bias_and_anharmonicity_rebind_use_the_new_calibration(self):
        """Grouped flux rebinding applies calibration parameters before moving the bias."""
        from quchip import Chip, FluxTunableTransmon

        q = FluxTunableTransmon(
            freq=4.47,
            anharmonicity=-0.2006,
            flux_bias=0.0,
            label="c",
        )
        chip = Chip([q])
        expected = chip.with_params({"c.anharmonicity": -0.24})["c"]
        expected.flux_bias = 0.3

        rebound = chip.with_params(
            {"c.flux_bias": 0.3, "c.anharmonicity": -0.24}
        )

        npt.assert_allclose(rebound.parameters["c.freq"], expected.freq, rtol=1e-10)
        npt.assert_allclose(
            float(rebound["c"]._E_J_max), float(expected._E_J_max), rtol=1e-10
        )

    def test_flux_bias_has_one_period_inverse_design_bounds(self):
        """Explicit flux optimization stays inside one canonical SQUID period."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006)

        assert q.tunable_param_bounds("flux_bias", 0.0) == (-0.5, 0.5)

    def test_flux_bias_rebind_round_trips_and_remains_jax_differentiable(self):
        """A rebound calibration serializes and its Hamiltonian remains differentiable in flux."""
        import jax
        import jax.numpy as jnp

        pytest.importorskip("dynamiqs")

        from quchip import Chip, FluxTunableTransmon
        from quchip.backend.dynamiqs import DynamiqsBackend

        q = FluxTunableTransmon(
            freq=4.47,
            anharmonicity=-0.2006,
            flux_bias=0.0,
            levels=3,
            label="c",
        )
        chip = Chip([q], backend=DynamiqsBackend())
        rebound = chip.with_params({"c.flux_bias": 0.2})
        restored = Chip.from_dict(rebound.to_dict())
        assert restored.parameters["c.flux_bias"] == pytest.approx(0.2)
        assert restored.parameters["c.freq"] == pytest.approx(rebound.parameters["c.freq"])

        def first_transition(flux_bias):
            shifted = chip.with_params({"c.flux_bias": flux_bias})
            matrix = shifted.hamiltonian().matrix(backend=shifted.backend)
            energies = jnp.linalg.eigvalsh(matrix)
            return energies[1] - energies[0]

        gradient = jax.jit(jax.grad(first_transition))(0.2)
        assert jnp.isfinite(gradient)
        assert abs(float(gradient)) > 1.0e-6


# ===========================================================================
# GaussianEdge envelope
# ===========================================================================


class TestGaussianEdge:


    def test_jax_traceable(self):
        """Envelope values should preserve JAX arrays."""
        import jax.numpy as jnp

        from quchip import GaussianEdge

        env = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=0.1)
        t = jnp.linspace(0.0, 80.0, 100)
        w = env.value(t)
        assert w.shape == (100,)


# ===========================================================================
# QuantumSequence.flux_to
# ===========================================================================


class TestFluxTo:
    def _make_chip(self):
        from quchip import Chip, FluxDrive, FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, label="QB")
        fdrv = FluxDrive(target=q, label="QB_z")
        chip = Chip([q], frame="rotating", backend="qutip")
        chip.wire(fdrv)
        return chip, q, fdrv


    def test_flux_to_does_not_mutate_envelope_template(self):
        """The original envelope object passed as envelope is not mutated."""
        from quchip import GaussianEdge, QuantumSequence

        chip, q, _ = self._make_chip()
        seq = QuantumSequence(chip)
        template = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=None)
        seq.flux_to(q, target_freq=4.0, envelope=template)
        assert template.amplitude is None
