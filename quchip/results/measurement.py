"""Captured physical field statistics and inexpensive IQ receiver processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from quchip.analysis.field_statistics import block_diagonal, quadrature_transfer
from quchip.results.input_output import MeanFieldResponseResult
from quchip.utils.jax_utils import array_namespace, contains_tracer, is_jax_array, select_array_module
from quchip.utils.labeling import resolve_label


def _capture(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.setflags(write=False)
    return value


def _index(ports: tuple[str, ...], output: Any) -> int:
    label = resolve_label(output)
    if label not in ports:
        raise KeyError(f"Unknown output {label!r}; available: {list(ports)}")
    return ports.index(label)


def _integrate(values: Any, frequencies: Any, xp: Any) -> Any:
    weights = xp.diff(frequencies)
    return xp.sum((values[..., 1:, :, :] + values[..., :-1, :, :]) * weights[:, None, None] / 2, axis=-3)


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


@dataclass(frozen=True)
class MeasurementSamples:
    """Synthetic Gaussian IQ draws with sample, sweep, then output axes."""

    ports: tuple[str, ...]
    input: str
    axes: tuple[tuple[str, Any], ...]
    incident: Any
    values: Any
    receiver: IQReceiver
    distribution: str = "Gaussian field-moment approximation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _capture(self.values))
        object.__setattr__(self, "incident", _capture(self.incident))

    def field(self, output: Any) -> Any:
        """Return sampled complex fields in 1/sqrt(ns)."""
        return self.values[..., _index(self.ports, output)]

    def ratio(self, output: Any) -> Any:
        """Return sampled output/input ratios, undefined for zero input."""
        xp = array_namespace(self.values)
        zero = self.incident == 0
        return xp.where(zero, xp.nan + 0j, self.field(output) / xp.where(zero, 1.0, self.incident))


@dataclass(frozen=True)
class MeasurementStatistics:
    """Integrated output means and full covariance of (I0,Q0,I1,Q1,...)."""

    ports: tuple[str, ...]
    input: str
    axes: tuple[tuple[str, Any], ...]
    incident: Any
    values: Any
    iq_covariance: Any
    receiver: IQReceiver
    contributions: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("incident", "values", "iq_covariance"):
            object.__setattr__(self, name, _capture(getattr(self, name)))
        object.__setattr__(self, "contributions", MappingProxyType(
            {key: _capture(value) for key, value in self.contributions.items()}))

    def mean(self, output: Any) -> Any:
        """Return the integrated complex mean at an output."""
        return self.values[..., _index(self.ports, output)]

    def covariance(self, output: Any, other: Any = None) -> Any:
        """Return a 2x2 IQ covariance or cross-output covariance block."""
        i = 2 * _index(self.ports, output)
        j = i if other is None else 2 * _index(self.ports, other)
        return self.iq_covariance[..., i:i+2, j:j+2]

    def noise_contributions(self, output: Any) -> Mapping[str, Any]:
        """Return integrated source covariance blocks, including detector vacuum."""
        i = 2 * _index(self.ports, output)
        return MappingProxyType({key: value[..., i:i+2, i:i+2] for key, value in self.contributions.items()})

    def sample(self, count: int, *, seed: int | None = None, key: Any = None) -> MeasurementSamples:
        """Draw joint Gaussian IQ samples without running a physical solver.

        Use seed for NumPy draws or an explicit JAX random key. The covariance
        includes anomalous/cross-output second moments, but does not specify
        higher-order non-Gaussian photon statistics.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Sample count must be a positive integer.")
        if seed is not None and key is not None:
            raise ValueError("Specify seed or key, not both.")
        xp = array_namespace(self.iq_covariance)
        if key is not None:
            import jax.numpy as xp
        shape = (count, *self.values.shape[:-1], 2 * len(self.ports))
        if key is None and contains_tracer(self.iq_covariance):
            raise ValueError("Sampling traced statistics requires an explicit JAX random key.")
        if key is None:
            standard = xp.asarray(np.random.default_rng(seed).normal(size=shape))
        else:
            import jax.random
            standard = jax.random.normal(key, shape, dtype=self.iq_covariance.dtype)
        root = xp.linalg.cholesky(xp.asarray(self.iq_covariance))
        fluctuations = xp.einsum("...ij,n...j->n...i", root, standard)
        values = self.values + fluctuations[..., 0::2] + 1j * fluctuations[..., 1::2]
        return MeasurementSamples(self.ports, self.input, self.axes, self.incident, values, self.receiver)

    def calibrate(self, factors: Mapping[Any, Any]) -> MeasurementStatistics:
        """Multiply output fields and both covariance axes by calibration factors."""
        xp = select_array_module(is_jax_array(self.values) or contains_tracer(tuple(factors.values())))
        normalized = {resolve_label(port): value for port, value in factors.items()}
        for port in normalized:
            _index(self.ports, port)
        gains = xp.asarray([normalized.get(port, 1.0) for port in self.ports])
        blocks = quadrature_transfer(gains, gains, xp)
        transform = block_diagonal(blocks, xp)
        transform = xp.real(transform)
        covariance = transform @ self.iq_covariance @ transform.T
        contributions = {name: transform @ value @ transform.T for name, value in self.contributions.items()}
        return MeasurementStatistics(self.ports, self.input, self.axes, self.incident,
                                     self.values * gains, covariance, self.receiver, contributions)


@dataclass(frozen=True)
class MeasurementResult(MeanFieldResponseResult):
    """Physical means and spectra captured before choosing a receiver.

    noise_components maps physical source labels to (white covariance,
    excess spectral covariance). The latter has an offset-frequency axis
    before its two IQ axes. Device-generated excess includes input-system
    correlations and can be negative; it is not an independent random source.
    """

    noise_frequencies: Any
    noise_components: Mapping[str, tuple[Any, Any]]
    output_delays: Any
    parameters: tuple[Mapping[str, Any], ...]
    conventions: tuple[str, ...] = (
        "b=I+iQ; frequencies in GHz; time in ns",
        "Physical normal-order spectra; detector vacuum added by the receiver",
        "Independent stationary experiments across sweep points",
        "Gaussian sampling uses second moments, not full photon statistics",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("values", "incident", "noise_frequencies", "frequencies", "amplitudes", "output_delays"):
            object.__setattr__(self, name, _capture(getattr(self, name)))
        object.__setattr__(self, "axes", tuple((name, _capture(value)) for name, value in self.axes))
        object.__setattr__(self, "parameters", tuple(MappingProxyType(
            {name: _capture(value) for name, value in point.items()}) for point in self.parameters))
        object.__setattr__(self, "noise_components", MappingProxyType({
            name: (_capture(white), _capture(excess)) for name, (white, excess) in self.noise_components.items()}))

    def noise_contributions(self, output: Any) -> Mapping[str, Any]:
        """Return physical IQ spectral contributions at the captured offsets."""
        i = 2 * _index(self.ports, output)
        return MappingProxyType({name: white[..., None, i:i+2, i:i+2] + excess[..., i:i+2, i:i+2]
                                 for name, (white, excess) in self.noise_components.items()})

    def statistics(self, *, receiver: IQReceiver) -> MeasurementStatistics:
        """Integrate captured spectra and detector vacuum without a solver call."""
        receiver_upper = None if receiver.transfer is None else receiver.transfer(self.noise_frequencies)
        xp = select_array_module(is_jax_array(self.values)
                                 or contains_tracer((receiver.integration_time, receiver_upper)))
        frequencies = xp.asarray(self.noise_frequencies)
        count = len(self.ports)
        delays = xp.repeat(self.output_delays, 2, axis=-1)
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
        components = dict(self.noise_components)
        components["receiver.vacuum"] = (xp.broadcast_to(xp.eye(2 * count) / 2,
                                                        (*self.shape, 2 * count, 2 * count)),
                                         xp.zeros((*self.shape, len(frequencies), 2 * count, 2 * count)))
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
                                 "measure on a finer spectral grid.")
            if sum(float(value) for value in edge_bounds) / scale > receiver.tolerance:
                raise ValueError("Receiver integration extends beyond captured spectral support; "
                                 "measure on a wider noise_frequencies grid.")
            if receiver.transfer is not None:
                edge = np.max(np.abs(np.asarray(receiver.transfer(frequencies[[0, -1]]))))
                if edge > receiver.tolerance:
                    raise ValueError("Receiver filter extends beyond captured spectral support; "
                                     "measure on a wider noise_frequencies grid.")
            if np.min(np.linalg.eigvalsh(np.asarray(covariance))) <= 0:
                raise ValueError("Integrated IQ covariance is not positive; "
                                 "check spectral resolution and model validity.")
        return MeasurementStatistics(self.ports, self.input, self.axes, self.incident,
                                     self.values * mean_gain, covariance, receiver, contributions)

    def sample(
        self, count: int, *, receiver: IQReceiver, seed: int | None = None, key: Any = None,
    ) -> MeasurementSamples:
        """Integrate for a receiver and draw samples from the captured moments."""
        return self.statistics(receiver=receiver).sample(count, seed=seed, key=key)
