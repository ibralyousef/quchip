"""Cartesian coordinates shared by static sweeps and scheduled batches."""

from collections.abc import Hashable, Sequence
from typing import Any, TypeVar

import numpy as np

Key = TypeVar("Key", bound=Hashable)


def expand_axis_groups(
    groups: Sequence[Sequence[dict[Key, Any]]],
) -> tuple[tuple[int, ...], list[tuple[tuple[int, ...], dict[Key, Any]]]]:
    """Expand validated independent groups, with zipped values already paired."""
    shape = tuple(len(group) for group in groups)
    points: list[tuple[tuple[int, ...], dict[Key, Any]]] = []
    for coord in np.ndindex(*shape):
        bindings: dict[Key, Any] = {}
        for group, index in zip(groups, coord, strict=True):
            bindings.update(group[index])
        points.append((coord, bindings))
    return shape, points
