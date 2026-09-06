"""Selection and interpolation over retained result coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np

from quchip.utils.jax_utils import contains_tracer


def require_valid(value: Any, invalid: Any, message: str) -> Any:
    """Keep coordinate validation active for concrete and traced queries."""
    if contains_tracer(invalid):
        import equinox as eqx

        return eqx.error_if(value, invalid, message)
    if np.any(invalid):
        raise ValueError(message)
    return value


def time_selection(times: Any, t: Any, method: str, xp: Any) -> tuple[Any, Any, Any]:
    """Return bracketing indices and linear weight, preserving query shape."""
    if method not in ("exact", "nearest", "interpolate"):
        raise ValueError('Lookup method must be "exact", "nearest", or "interpolate".')
    times, query = xp.asarray(times), xp.asarray(t)
    if xp.iscomplexobj(query):
        raise ValueError("Lookup times must be real.")
    query = require_valid(query, xp.any(~xp.isfinite(query) | (query < times[0]) | (query > times[-1])),
                          "Lookup times must be finite and lie in the simulated interval.")
    right = xp.minimum(xp.searchsorted(times, query, side="left"), len(times) - 1)
    left = xp.maximum(right - 1, 0)
    if method == "exact":
        right = require_valid(right, xp.any(times[right] != query),
                              'No saved time equals the query; select a saved time or use method="nearest" explicitly.')
        return right, right, xp.zeros_like(query)
    if method == "nearest":
        index = xp.where(query - times[left] <= times[right] - query, left, right)
        return index, index, xp.zeros_like(query)
    width = times[right] - times[left]
    weight = (query - times[left]) / xp.where(width == 0, 1, width)
    return left, right, weight


def observable_at(times: Any, t: Any, values: Any, method: str, xp: Any) -> Any:
    """Query a last-axis time series, retaining native real or complex values."""
    values = xp.asarray(values)
    if values.ndim == 0 or values.shape[-1] != len(times):
        raise ValueError("Observable values must have len(result.times) entries on their last axis.")
    left, right, weight = time_selection(times, t, method, xp)
    lo = xp.take(values, left, axis=-1)
    if method != "interpolate":
        return lo
    hi = xp.take(values, right, axis=-1)
    return (1 - weight) * lo + weight * hi
