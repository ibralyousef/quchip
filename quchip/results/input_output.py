"""Results for continuous-wave port scattering calculations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class SParameterResult:
    """Complete selected-plane small-signal scattering over a sweep grid.

    ``matrix`` has shape ``(*shape, n_planes, n_planes)`` and is indexed
    ``[..., output, input]`` in ``planes`` order. ``numpy.asarray(result)``
    returns ``matrix``.
    """

    frequencies: Any
    planes: tuple[str, ...]
    axes: tuple[tuple[str, Any], ...]
    shape: tuple[int, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    matrix: Any

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
class OutputSpectrumResult:
    """Normally ordered stationary output-field fluctuation spectrum."""

    port: str
    frequencies: Any
    fluctuation_spectrum: Any
    output_photon_flux: Any
    coherent_flux: Any
    incoherent_flux: Any
    steady_state: Any
    added_noise_spectrum: Any
    fourier_convention: str = "2 Re integral_0^inf d tau exp(+i 2 pi f tau) C(tau)"

    @property
    def total_flux(self) -> Any:
        """Mean normally ordered output photon flux."""
        return self.output_photon_flux


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
