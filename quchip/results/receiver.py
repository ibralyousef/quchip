"""Shared heterodyne receiver integration and random sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from quchip.analysis.field_statistics import quadrature_transfer
from quchip.utils.jax_utils import contains_tracer, is_jax_array, select_array_module


@dataclass(frozen=True)
class IQReceiver:
    """Configure ideal heterodyne detection after the physical calculation.

    integration_time : scalar
        Boxcar integration duration in ns; white noise scales as 1/T.
    transfer : callable, optional
        Additional digital amplitude transfer at offset frequencies in GHz.
        It must be covered by the measurement's stored spectral grid. DC gain
        also transforms the mean. No filter is applied to the physical chip.
    tolerance : float
        Relative tolerance for quadrature-grid convergence checks.
    """

    integration_time: Any
    transfer: Callable[[Any], Any] | None = field(default=None, repr=False)
    tolerance: float = 0.02

    def __post_init__(self) -> None:
        if not contains_tracer(self.integration_time):
            value = np.asarray(self.integration_time)
            if value.ndim or not np.isfinite(value) or value <= 0:
                raise ValueError("integration_time must be a finite positive scalar in ns.")
        if self.transfer is not None and not callable(self.transfer):
            raise TypeError("Receiver transfer must be callable.")
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError("Receiver tolerance must lie between zero and one.")


def validate_samples(count: int, seed: Any, key: Any, values: Any) -> Any:
    """Validate shot configuration and choose its native array namespace."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Sample count must be a positive integer.")
    if seed is not None and key is not None:
        raise ValueError("Specify seed or key, not both.")
    if key is None and contains_tracer(values):
        raise ValueError("Sampling traced statistics requires an explicit JAX random key.")
    if key is not None:
        import jax.numpy as jnp
        return jnp
    from jax.tree_util import tree_leaves
    return select_array_module(any(is_jax_array(v) for v in tree_leaves(values)))


def gaussian_samples(mean: Any, covariance: Any, count: int, *, seed: Any = None, key: Any = None) -> Any:
    """Draw real Gaussian vectors with a leading shot axis and full covariance."""
    xp = validate_samples(count, seed, key, (mean, covariance))
    mean, covariance = xp.asarray(mean), xp.asarray(covariance)
    shape = (count, *mean.shape)
    if key is None:
        standard = xp.asarray(np.random.default_rng(seed).normal(size=shape))
    else:
        import jax.random
        standard = jax.random.normal(key, shape, dtype=covariance.dtype)
    root = xp.linalg.cholesky(covariance)
    return mean + xp.einsum("...ij,n...j->n...i", root, standard)


def noise_grid(frequencies: Any = None) -> np.ndarray:
    """Validate a two-sided offset grid in GHz, or supply the standard grid."""
    if frequencies is None:
        positive = np.geomspace(1e-9, 0.1, 161)
        return np.concatenate((-positive[::-1], [0.0], positive))
    values = np.asarray(frequencies, dtype=float)
    if (values.ndim != 1 or len(values) < 5 or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0) or not np.any(values == 0)
            or not np.allclose(values, -values[::-1], atol=0, rtol=1e-12)):
        raise ValueError("noise_frequencies must be finite, increasing, symmetric offsets including zero (GHz).")
    return values


def _integrate(values: Any, frequencies: Any, xp: Any) -> Any:
    weights = xp.diff(frequencies)
    return xp.sum((values[..., 1:, :, :] + values[..., :-1, :, :]) * weights[:, None, None] / 2, axis=-3)


def integrate_noise(values: Any, noise_frequencies: Any, noise_components: Any,
                    output_delays: Any, receiver: IQReceiver) -> tuple[Any, Any, Any]:
    """Integrate normal spectra plus detector vacuum; return covariance, budget, DC gain."""
    count = values.shape[-1]
    receiver_upper = None if receiver.transfer is None else receiver.transfer(noise_frequencies)
    xp = select_array_module(is_jax_array(values)
                             or contains_tracer((receiver.integration_time, receiver_upper)))
    frequencies = xp.asarray(noise_frequencies)
    delays = xp.repeat(output_delays, 2, axis=-1)
    overlap = xp.maximum(1 - xp.abs(delays[..., :, None] - delays[..., None, :]) / receiver.integration_time, 0)
    window = xp.sinc(frequencies * receiver.integration_time) ** 2
    mean_gain = 1.0
    transform = None
    if receiver.transfer is not None:
        upper = xp.asarray(receiver_upper) + xp.zeros_like(frequencies)
        lower = xp.asarray(receiver.transfer(-frequencies)) + xp.zeros_like(frequencies)
        mean_gain = receiver.transfer(xp.asarray(0.0))
        local = quadrature_transfer(upper, lower, xp)
        transform = xp.stack([xp.kron(xp.eye(count), block) for block in local])
    components = dict(noise_components)
    components["receiver.vacuum"] = (xp.broadcast_to(xp.eye(2 * count) / 2,
                                                    (*values.shape[:-1], 2 * count, 2 * count)),
                                     xp.zeros((*values.shape[:-1], len(frequencies), 2 * count, 2 * count)))
    contributions, coarse, edge_bounds = {}, [], []
    for name, (white, excess) in components.items():
        white, excess = xp.asarray(white), xp.asarray(excess)
        if transform is None:
            spectrum = excess
            analytic = white * overlap / receiver.integration_time
        else:
            relative = delays[..., :, None] - delays[..., None, :]
            phase = xp.exp(2j * xp.pi * frequencies[:, None, None] * relative[..., None, :, :])
            physical = excess + white[..., None, :, :] * phase
            spectrum = transform @ physical @ xp.conj(xp.swapaxes(transform, -1, -2))
            analytic = 0.0
        edge_bounds.append(xp.max(xp.abs(spectrum[..., xp.asarray([0, -1]), :, :]))
                           / (xp.pi**2 * frequencies[-1] * receiver.integration_time**2))
        weighted = xp.real(spectrum) * window[:, None, None]
        contributions[name] = analytic + _integrate(weighted, frequencies, xp)
        coarse.append(analytic + _integrate(weighted[..., ::2, :, :], frequencies[::2], xp))
    covariance = sum(contributions.values())
    covariance = (covariance + xp.swapaxes(covariance, -1, -2)) / 2
    if not contains_tracer((covariance, receiver.integration_time)):
        scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
        error = float(np.max(np.abs(np.asarray(covariance - sum(coarse))))) / scale
        if error > receiver.tolerance:
            raise ValueError("Receiver integration is unresolved on the captured noise_frequencies; "
                             "capture a finer spectral grid.")
        if sum(float(value) for value in edge_bounds) / scale > receiver.tolerance:
            raise ValueError("Receiver integration extends beyond captured spectral support; "
                             "capture a wider noise_frequencies grid.")
        if receiver.transfer is not None:
            edge = np.max(np.abs(np.asarray(receiver.transfer(frequencies[[0, -1]]))))
            if edge > receiver.tolerance:
                raise ValueError("Receiver filter extends beyond captured spectral support; "
                                 "capture a wider noise_frequencies grid.")
        if np.min(np.linalg.eigvalsh(np.asarray(covariance))) <= 0:
            raise ValueError("Integrated IQ covariance is not positive; "
                             "check spectral resolution and model validity.")
    return covariance, contributions, mean_gain
