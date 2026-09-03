"""Results for continuous-wave port scattering calculations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quchip.utils.jax_utils import is_jax_array, select_array_module

from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class SParameterResult:
    """Complete selected-plane small-signal scattering over a sweep grid.

    Around a phase-sensitive operating point, the response is
    ``delta <b_out> = S delta beta + T conj(delta beta)``. ``matrix`` stores ``S``
    and ``conjugate_matrix`` stores ``T``; both have shape ``(*shape, n_planes,
    n_planes)`` and use ``[..., output, input]`` indexing in ``planes`` order.
    ``s(output, input)`` and ``t(output, input)`` select individual entries.

    The stationary route computes both matrices from one shifted-Liouvillian
    factorization. The passive-linear route reports zero for ``T``.
    ``numpy.asarray(result)`` returns ``matrix``.
    """

    frequencies: Any
    planes: tuple[str, ...]
    axes: tuple[tuple[str, Any], ...]
    shape: tuple[int, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    matrix: Any
    conjugate_matrix: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            tuple(MappingProxyType(dict(item)) for item in self.diagnostics),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of the result axes, in array order."""
        return tuple(name for name, _ in self.axes)

    def s(self, output: Any, input: Any) -> Any:
        """Return ``S(output, input)`` over the sweep grid for two selected planes."""
        return self.matrix[..., self._index(output), self._index(input)]

    def t(self, output: Any, input: Any) -> Any:
        """Return the phase-conjugating ``T(output, input)`` over the sweep grid."""
        return self.conjugate_matrix[..., self._index(output), self._index(input)]

    @property
    def s11(self) -> Any:
        """Return reflection from the first selected plane back onto itself."""
        return self.matrix[..., 0, 0]

    @property
    def s21(self) -> Any:
        """Return transmission from the first selected plane to the second."""
        if len(self.planes) < 2:
            raise AttributeError("s21 requires at least two planes.")
        return self.matrix[..., 1, 0]

    def _index(self, plane: Any) -> int:
        label = resolve_label(plane)
        try:
            return self.planes.index(label)
        except ValueError:
            raise KeyError(f"Plane {label!r} is not in this result. Available: {list(self.planes)}") from None

    def __array__(self) -> np.ndarray:
        return np.asarray(self.matrix)


@dataclass(frozen=True)
class MeanFieldResponseResult:
    """Stationary mean output fields from a finite coherent probe.

    ``values`` stores ``<b_out>`` at every selected plane with shape
    ``(*shape, n_planes)`` in ``planes`` order. ``incident`` stores the input
    amplitude ``beta`` broadcast to ``shape``. ``axes`` lists chip and pump sweep
    axes first, followed by ``"amplitude"`` and ``"frequency"`` when those
    arguments are arrays.

    ``ratio(plane)`` approaches the corresponding small-signal S-parameter as
    ``beta`` tends to zero when no fixed pump leaves a coherent mean at that
    plane and carrier, and is ``NaN`` where ``beta`` is zero. The result
    contains one stationary mean-field branch; it does not encode sweep-rate
    hysteresis or metastable branches.
    """

    planes: tuple[str, ...]
    input: str
    frequencies: Any
    amplitudes: Any
    axes: tuple[tuple[str, Any], ...]
    shape: tuple[int, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    values: Any
    incident: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            tuple(MappingProxyType(dict(item)) for item in self.diagnostics),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of the result axes, in array order."""
        return tuple(name for name, _ in self.axes)

    def mean(self, plane: Any) -> Any:
        """Return the stationary ``<b_out>`` at ``plane`` with shape ``shape``."""
        label = resolve_label(plane)
        try:
            return self.values[..., self.planes.index(label)]
        except ValueError:
            raise KeyError(f"Plane {label!r} is not in this result. Available: {list(self.planes)}") from None

    def ratio(self, plane: Any) -> Any:
        """Return the stationary ``<b_out>/beta`` at ``plane`` with shape ``shape``.

        The result is complex ``NaN`` where the incident ``beta`` is zero. Its
        zero-amplitude limit is the corresponding small-signal S-parameter when
        no fixed pump leaves a coherent mean at that plane and carrier.
        """
        mean = self.mean(plane)
        xp = select_array_module(is_jax_array(mean))
        incident = xp.asarray(self.incident)
        zero = incident == 0
        return xp.where(zero, xp.nan + 0j, mean / xp.where(zero, 1.0, incident))


@dataclass(frozen=True)
class OutputSpectrumResult:
    """Stationary output-field fluctuation spectra and signal photon fluxes.

    ``signal_fluctuation_spectrum`` is the normally ordered signal spectral
    density. ``added_noise_spectrum`` is the amplifier added-noise density,
    and ``total_fluctuation_spectrum`` is their sum. ``signal_photon_flux``
    is the signal's mean normally ordered flux, split into
    ``signal_coherent_flux`` and ``signal_incoherent_flux``. Added noise is
    not included in these fluxes because converting a spectral density to
    flux requires a detection bandwidth.
    """

    port: str
    frequencies: Any
    total_fluctuation_spectrum: Any
    signal_fluctuation_spectrum: Any
    added_noise_spectrum: Any
    signal_photon_flux: Any
    signal_coherent_flux: Any
    signal_incoherent_flux: Any
    steady_state: Any
    fourier_convention: str = "2 Re integral_0^inf d tau exp(+i 2 pi f tau) C(tau)"


@dataclass(frozen=True)
class OutputCorrelationResult:
    """Normalized stationary output-field correlation versus delay."""

    order: int
    input_port: str
    output_port: str
    delays: Any
    values: Any
    unnormalized: Any
    input_intensity: Any
    output_intensity: Any
    steady_state: Any
    normalization: str

    @property
    def port(self) -> str:
        """Delayed output port, retained for single-port result code."""
        return self.output_port

    @property
    def intensity(self) -> Any:
        """Delayed output intensity, retained for single-port result code."""
        return self.output_intensity
