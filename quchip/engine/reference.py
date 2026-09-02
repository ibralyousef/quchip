"""Reference-plane transforms applied outside the Markovian core."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, select_array_module


@dataclass(frozen=True)
class ReferenceDelay:
    """One reference section with propagation duration ``duration`` ns."""

    label: str
    duration: Any


@dataclass(frozen=True)
class ReferenceFilter:
    """One passive filter reference section and its tracked parameters.

    ``transfer(frequency, **parameters)`` accepts a scalar or array frequency in
    GHz and returns a complex amplitude transfer.
    """

    label: str
    transfer: Callable[..., Any]
    parameters: Mapping[str, Any]

    def __call__(self, frequency: Any) -> Any:
        """Evaluate the complex transfer at ``frequency`` in GHz.

        ``frequency`` may be scalar or array-valued. Reject a concrete result whose
        magnitude exceeds one; do not concretize traced results for this check.
        """
        value = self.transfer(frequency, **self.parameters)
        if not contains_tracer(value) and np.any(np.abs(np.asarray(value)) > 1.0 + 1e-12):
            raise ValueError(
                f"Filter {self.label!r} must be passive, but its evaluated transfer has |H| > 1. "
                "Gain requires an amplifier section."
            )
        return value


ReferenceElement = ReferenceDelay | ReferenceFilter


@dataclass(frozen=True)
class ReferencePlane:
    """Inbound and outbound reference runs for one exposure, in propagation order."""

    inbound: tuple[ReferenceElement, ...] = ()
    outbound: tuple[ReferenceElement, ...] = ()


def _array_module(elements: tuple[ReferenceElement, ...], frequency: Any, xp: Any) -> Any:
    """Return ``xp``, or the module matching traced content in ``elements`` and ``frequency``."""
    if xp is not None:
        return xp
    values: list[Any] = [frequency]
    for element in elements:
        if isinstance(element, ReferenceDelay):
            values.append(element.duration)
        else:
            values.extend(element.parameters.values())
    return select_array_module(contains_tracer(tuple(values)))


def cw_transfer(elements: tuple[ReferenceElement, ...], frequency: Any, xp: Any = None) -> Any:
    """Return the exact continuous-wave transfer of one reference leg.

    At ``frequency`` in GHz, each delay contributes
    ``exp(+i 2π frequency duration)`` and each filter contributes
    ``H(frequency)``. Delay durations are in ns.
    """
    xp = _array_module(elements, frequency, xp)
    transfer = xp.asarray(1.0 + 0.0j)
    for element in elements:
        if isinstance(element, ReferenceDelay):
            transfer = transfer * xp.exp(
                1j * TWO_PI * xp.asarray(frequency) * xp.asarray(element.duration)
            )
        else:
            transfer = transfer * xp.asarray(element(frequency))
    return transfer


def carrier_transfer(elements: tuple[ReferenceElement, ...], carrier: Any, xp: Any = None) -> Any:
    """Return the narrowband filter transfer of one reference leg.

    Evaluate every filter at ``carrier`` in GHz. Ignore delays because transient
    propagation applies them as time shifts.
    """
    xp = _array_module(elements, carrier, xp)
    transfer = xp.asarray(1.0 + 0.0j)
    for element in elements:
        if isinstance(element, ReferenceFilter):
            transfer = transfer * xp.asarray(element(carrier))
    return transfer


def has_filter(elements: tuple[ReferenceElement, ...]) -> bool:
    """Return whether a reference leg contains a filter section."""
    return any(isinstance(element, ReferenceFilter) for element in elements)


def time_shift(elements: tuple[ReferenceElement, ...]) -> Any:
    """Return the summed delay duration of one reference leg, in ns."""
    return sum((element.duration for element in elements if isinstance(element, ReferenceDelay)), 0.0)
