"""Reference-plane transforms applied outside the Markovian core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quchip.utils.constants import TWO_PI


@dataclass(frozen=True)
class ReferenceDelay:
    """One reference section with propagation duration ``duration`` ns."""

    label: str
    duration: Any


@dataclass(frozen=True)
class ReferencePlane:
    """Inbound and outbound reference runs for one exposure, in propagation order."""

    inbound: tuple[ReferenceDelay, ...] = ()
    outbound: tuple[ReferenceDelay, ...] = ()


def cw_transfer(elements: tuple[ReferenceDelay, ...], frequency: Any, xp: Any) -> Any:
    """Return the CW leg factor ``exp(+i 2π f τ)``.

    ``frequency`` is in GHz; ``τ`` is the total duration of ``elements`` in ns.
    """
    transfer = xp.asarray(1.0 + 0.0j)
    for element in elements:
        transfer = transfer * xp.exp(
            1j * TWO_PI * xp.asarray(frequency) * xp.asarray(element.duration)
        )
    return transfer


def time_shift(elements: tuple[ReferenceDelay, ...]) -> Any:
    """Return the total duration of one reference leg in ns."""
    return sum((element.duration for element in elements), 0.0)
