"""Component labels and object-or-label lookup.

Components without an explicit label receive ``"{prefix}_{n}"``, with a
process-wide counter per prefix starting at zero. Labels identify components
within a chip; :func:`resolve_label` accepts either the component or its label.
Use :func:`reset_label_counters` for deterministic labels in tests.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

_label_counters: dict[str, int] = {}


def auto_label(prefix: str) -> str:
    """Return the next ``"{prefix}_{n}"`` label; each prefix starts at zero."""
    idx = _label_counters.get(prefix, 0)
    _label_counters[prefix] = idx + 1
    return f"{prefix}_{idx}"


def reset_label_counters() -> None:
    """Reset the process-wide label counters, typically in test fixtures."""
    _label_counters.clear()


def resolve_label(obj: str | Any) -> str:
    """Return a string label from a string or labeled component.

    Raise ``TypeError`` if the object has no usable label.
    """
    if isinstance(obj, str):
        return obj
    label = getattr(obj, "label", None)
    if label is None:
        raise TypeError(
            f"Expected a label string or an object with .label, got {type(obj).__name__}: {obj!r}"
        )
    return str(label)


def merge_labeled_values(
    mapping: Mapping[Any, Any] | None,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge object-or-label keys and keyword arguments into a label-keyed dict.

    Duplicate labels, including a label supplied through both inputs, raise
    ``ValueError``. Values are unchanged; callers validate their types and bounds.
    """
    merged: dict[str, Any] = {}
    if mapping is not None:
        for key, value in mapping.items():
            label = resolve_label(key)
            if label in merged:
                raise ValueError(f"Duplicate device specification for '{label}'")
            merged[label] = value
    for label, value in kwargs.items():
        if label in merged:
            raise ValueError(f"Duplicate device specification for '{label}'")
        merged[label] = value
    return merged


def bare_label_from_mapping(
    device_labels: Sequence[str],
    mapping: Mapping[Any, Any] | None,
    kwargs: Mapping[str, Any],
) -> tuple[int, ...]:
    """Build a bare-state tuple in ``device_labels`` order from a partial mapping.

    Keys may be devices or labels; unspecified devices default to Fock index zero.
    Unknown or duplicate labels raise ``ValueError``. Callers validate values.
    """
    merged = merge_labeled_values(mapping, kwargs)

    unknown = sorted(set(merged) - set(device_labels))
    if unknown:
        raise ValueError(f"Unknown device labels {unknown}. Available labels: {list(device_labels)}")

    return tuple(merged.get(label, 0) for label in device_labels)


class LabelKeyedDict(dict):
    """Label-keyed result mapping that also accepts component objects.

    Two-element tuple keys match in either order. Iteration and serialization expose the
    stored keys.
    """

    @staticmethod
    def _canonical(key: Any) -> Any:
        """Resolve *key* (or each element of a tuple key) to its label form."""
        try:
            if isinstance(key, tuple):
                return tuple(resolve_label(part) for part in key)
            return resolve_label(key)
        except TypeError:
            return key  # keys that are neither labels nor labeled objects pass through

    def __getitem__(self, key: Any) -> Any:
        """Look up *key*, falling back to a 2-tuple key's reversal when the forward order misses."""
        resolved = self._canonical(key)
        if not super().__contains__(resolved) and isinstance(resolved, tuple) and len(resolved) == 2:
            reordered = resolved[::-1]
            if super().__contains__(reordered):
                resolved = reordered
        return super().__getitem__(resolved)

    def __contains__(self, key: Any) -> bool:
        """Report membership, matching a 2-tuple key's reversal when the forward order misses."""
        resolved = self._canonical(key)
        if super().__contains__(resolved):
            return True
        return isinstance(resolved, tuple) and len(resolved) == 2 and super().__contains__(resolved[::-1])

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent (via :meth:`__getitem__`)."""
        try:
            return self[key]
        except KeyError:
            return default


def top_components(
    eigenvector_matrix: Any,
    bare_labels: Sequence[tuple[int, ...]],
    dressed_idx: int,
    n: int,
) -> dict[tuple[int, ...], float]:
    """Return the top ``n`` bare-basis probabilities of a dressed eigenvector.

    Squared amplitudes from column ``dressed_idx`` are paired with bare labels
    in descending order. Requires a concrete eigenvector matrix.
    """
    probs = np.asarray(np.abs(eigenvector_matrix[:, dressed_idx]) ** 2, dtype=float)
    order = np.argsort(probs)[::-1][:n]
    return {bare_labels[idx]: float(probs[idx]) for idx in order}
