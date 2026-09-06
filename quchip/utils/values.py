"""Ownership and cache keys for authored numerical values."""

from __future__ import annotations

import copy
from typing import Any, Callable
from dataclasses import dataclass, field

import jax
import jax.tree_util as jtu
import numpy as np
from jax.core import Tracer


def copy_value(value: Any, *, readonly: bool = False) -> Any:
    """Copy mutable values without detaching native JAX arrays or tracers."""
    memo: dict[int, Any] = {}

    def capture(item: Any) -> Any:
        if id(item) in memo:
            return memo[id(item)]
        if isinstance(item, (jax.Array, Tracer)):
            memo[id(item)] = item
            return item
        if isinstance(item, np.ndarray):
            copied = copy.deepcopy(item, memo)
            copied.flags.writeable = not readonly
            return copied
        if isinstance(item, dict):
            copied_dict: dict[Any, Any] = {}
            memo[id(item)] = copied_dict
            copied_dict.update((key, capture(value)) for key, value in item.items())
            return copied_dict
        if isinstance(item, list):
            copied_list: list[Any] = []
            memo[id(item)] = copied_list
            copied_list.extend(capture(value) for value in item)
            return copied_list
        # Seed deepcopy's memo from registered trees, including immutable JAX
        # leaves inside native backend objects and extension payloads.
        for leaf in jtu.tree_leaves(item, is_leaf=lambda child: child is not item):
            if leaf is not item:
                memo[id(leaf)] = capture(leaf)
        return copy.deepcopy(item, memo)

    return capture(value)


def value_fingerprint(value: Any) -> Any:
    """Hash supported value contents; reject traced or opaque mutable payloads."""
    if isinstance(value, Tracer):
        raise ValueError("Traced values cannot key an eager calculation cache.")
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex, type)):
        return value
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Object arrays cannot key a calculation cache.")
        return array.shape, array.dtype.str, array.tobytes()
    if isinstance(value, dict):
        return tuple((value_fingerprint(key), value_fingerprint(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return type(value), tuple(value_fingerprint(item) for item in value)
    leaves, tree = jtu.tree_flatten(value)
    if len(leaves) == 1 and leaves[0] is value:
        raise ValueError(f"Opaque {type(value).__name__} cannot key a calculation cache.")

    def structure(node: Any) -> Any:
        data = node.node_data()
        return value_fingerprint(data), tuple(structure(child) for child in node.children())

    return structure(tree), tuple(value_fingerprint(leaf) for leaf in leaves)


@dataclass(eq=False)
class DeferredValue:
    """Evaluate captured inputs on demand; cache only concrete values."""

    evaluate: Callable[[], Any] = field(repr=False)
    _value: Any = field(default=None, init=False, repr=False)

    def __call__(self) -> Any:
        from quchip.utils.jax_utils import contains_tracer

        if self._value is None:
            value = self.evaluate()
            if not contains_tracer(value):
                self._value = value
            return value
        return self._value
