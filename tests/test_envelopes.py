"""Unit tests for pulse envelope waveform generation.

Analytical envelope definitions determine the reference waveforms; no backend
or chip is required.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.control.envelopes import Gaussian, GaussianDRAG, Square, SquareWithGaussianEdges


def test_square_rejects_envelope_owned_global_phase():
    """Global phase belongs to scheduling, not the local square shape."""
    with pytest.raises(TypeError, match="phase"):
        Square(duration=10.0, amplitude=0.5, phase=0.2)


class TestSquare:
    """Tests for the constant-amplitude Square envelope."""

    def test_constant_amplitude(self):
        """All samples should equal the given amplitude (magnitude)."""
        amp = 2.5
        sq = Square(duration=50.0, amplitude=amp)
        t = np.linspace(0, 50.0, 100)
        w = sq.value(t)
        npt.assert_allclose(np.abs(w), amp, atol=1e-14)

    def test_defaults_all_real_ones(self):
        """Default amplitude produces an all-real, all-ones local shape."""
        sq = Square(duration=10.0)
        t = np.linspace(0, 10.0, 50)
        w = sq.value(t)
        npt.assert_allclose(w.real, 1.0, atol=1e-14)
        npt.assert_allclose(w.imag, 0.0, atol=1e-14)

class TestGaussian:
    """Tests for the Gaussian window envelope."""

    def test_symmetry(self):
        """Waveform is approximately symmetric: |w[k]| ≈ |w[N-1-k]|."""
        # Integer-indexed sampling gives |k| and |N-1-k| centre-distances
        # differing by 1 sample; worst-case asymmetry at 1000 samples is ~1.8%.
        g = Gaussian(duration=100.0, amplitude=1.0)
        N = 1000
        t = np.linspace(0, 100.0, N)
        w = g.value(t)
        mag = np.abs(w)
        npt.assert_allclose(mag, mag[::-1], rtol=0.02)

    def test_edge_value_analytical(self):
        """The edge-to-center ratio should equal exp(-sigmas² / 2)."""
        sigmas = 3.0
        amp = 1.0
        g = Gaussian(duration=50.0, sigmas=sigmas, amplitude=amp)
        edge = np.abs(g.value(np.asarray([0.0]))[0])
        peak = np.abs(g.value(np.asarray([25.0]))[0])
        expected_ratio = np.exp(-(sigmas**2) / 2)  # exp(-4.5)
        npt.assert_allclose(edge / peak, expected_ratio, rtol=1e-6)

    def test_amplitude_scaling(self):
        """The waveform magnitude at the center should equal the amplitude."""
        amp = 5.0
        g = Gaussian(duration=100.0, amplitude=amp)
        center = np.abs(g.value(np.asarray([50.0]))[0])
        npt.assert_allclose(center, amp, rtol=1e-10)

class TestSquareWithGaussianEdges:
    """Tests for the flat-top pulse with Gaussian ramp edges."""

    def test_plateau_value(self):
        """Mid-pulse samples (plateau) should equal the amplitude."""
        env = SquareWithGaussianEdges(duration=40.0, amplitude=0.7, edge_frac=0.25)
        mid = np.abs(env.value(np.asarray([20.0]))[0])
        npt.assert_allclose(mid, 0.7, atol=1e-14)

    def test_ramp_endpoints(self):
        """At t = edge, value is at the peak amplitude; at t = 0, attenuated by exp(-(2*sigmas)^2/2)."""
        # sigma = edge / (2 N_σ), so the boundary (t=0) sits 2 N_σ sigmas from the ramp peak (t=edge).
        sigmas = 3.0
        env = SquareWithGaussianEdges(duration=40.0, amplitude=1.0, edge_frac=0.25, sigmas=sigmas)
        peak = np.abs(env.value(np.asarray([10.0]))[0])
        start = np.abs(env.value(np.asarray([0.0]))[0])
        npt.assert_allclose(peak, 1.0, atol=1e-14)
        npt.assert_allclose(start, np.exp(-((2 * sigmas) ** 2) / 2), rtol=1e-6)

    def test_edge_frac_rejects_out_of_range(self):
        """edge_frac must be in (0, 0.5]."""
        with pytest.raises(ValueError):
            SquareWithGaussianEdges(duration=10.0, amplitude=1.0, edge_frac=0.0)
        with pytest.raises(ValueError):
            SquareWithGaussianEdges(duration=10.0, amplitude=1.0, edge_frac=0.6)

    def test_shape_invariant_under_duration_rescale(self):
        """At matching fractional positions, the waveform is the same."""
        a = SquareWithGaussianEdges(duration=40.0, amplitude=1.0, edge_frac=0.25)
        b = SquareWithGaussianEdges(duration=100.0, amplitude=1.0, edge_frac=0.25)
        fracs = np.linspace(0.0, 1.0, 11)
        wa = a.value(fracs * a.duration)
        wb = b.value(fracs * b.duration)
        npt.assert_allclose(wa, wb, atol=1e-14)

    def test_roundtrip_serialization(self):
        """to_dict()/from_dict() round-trip preserves the waveform."""
        env = SquareWithGaussianEdges(duration=30.0, amplitude=0.5, edge_frac=0.3, sigmas=2.5)
        d = env.to_dict()
        restored = SquareWithGaussianEdges.from_dict(d)
        t = np.linspace(0, 30.0, 50)
        npt.assert_allclose(env.value(t), restored.value(t), atol=1e-14)


class TestSample:
    """Verify ``sample(tlist)`` evaluates ``value(tlist)`` elementwise."""

    def test_sample_matches_value_for_each_subclass(self):
        """sample(tlist) matches value(tlist) for every envelope subclass."""
        cases = [
            Square(duration=20.0, amplitude=0.7),
            Gaussian(duration=20.0, amplitude=0.5, sigmas=2.5),
            SquareWithGaussianEdges(duration=20.0, amplitude=0.6, edge_frac=0.2),
        ]
        t = np.linspace(0, 20.0, 51)
        for env in cases:
            npt.assert_allclose(env.sample(t), env.value(t), atol=1e-14)

    def test_sample_real_flag(self):
        """sample(..., real=True) returns the I part of a complex shape."""
        env = GaussianDRAG(duration=10.0, amplitude=1.2, beta=0.4)
        t = np.linspace(0, 10.0, 30)
        w = env.sample(t)
        r = env.sample(t, real=True)
        npt.assert_allclose(r, w.real, atol=1e-14)

    def test_sample_jax_traced(self):
        """Traced JAX input stays JAX-shaped; no Python-float concretization."""
        import jax
        import jax.numpy as jnp

        env = SquareWithGaussianEdges(duration=20.0, amplitude=0.5, edge_frac=0.25)

        @jax.jit
        def energy(tlist):
            # sum of |E(t)|^2, keeps the pipeline traced end-to-end.
            w = env.sample(tlist)
            return jnp.sum(jnp.abs(w) ** 2)

        t = jnp.linspace(0.0, 20.0, 41)
        val = float(energy(t))
        assert val > 0.0
