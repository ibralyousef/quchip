"""Captured effective Hamiltonian and Lindblad terms on a retained product space."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any

import jax.numpy as jnp
import numpy as np

from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.expr import PhysicsExpr
from quchip.utils.jax_utils import contains_tracer
from quchip.utils.values import copy_value, value_fingerprint


@dataclass(frozen=True, eq=False)
class OperatorProjection:
    """Captured authored coordinates for operators of surviving components.

    The rectangular embedding maps target authored coordinates into the source
    product space. Rates and local operators remain owned by the components;
    this record carries only their change of coordinates.
    """

    source_labels: tuple[str, ...]
    source_dims: tuple[int, ...]
    target_labels: tuple[str, ...]
    target_dims: tuple[int, ...]
    embedding: Any
    overrides: tuple[tuple[str, OperatorProjection], ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_labels", "source_dims", "target_labels", "target_dims"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for labels, dims in ((self.source_labels, self.source_dims), (self.target_labels, self.target_dims)):
            if len(labels) != len(dims) or len(set(labels)) != len(labels) or not labels:
                raise ValueError("OperatorProjection needs unique labels and matching dimensions.")
            if any(not isinstance(label, str) or not label for label in labels):
                raise ValueError("OperatorProjection labels must be nonempty strings.")
            if any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in dims):
                raise ValueError("OperatorProjection dimensions must be positive integers.")
        if not set(self.target_labels) <= set(self.source_labels):
            raise ValueError("Projected devices must belong to the source space.")
        matrix = copy_value(self.embedding, readonly=True)
        if getattr(matrix, "shape", None) != (prod(self.source_dims), prod(self.target_dims)):
            raise ValueError("OperatorProjection embedding has incompatible dimensions.")
        if not contains_tracer(matrix) and not np.all(np.isfinite(matrix)):
            raise ValueError("OperatorProjection embedding must be finite.")
        object.__setattr__(self, "embedding", matrix)
        overrides = tuple(self.overrides)
        if len({key for key, _ in overrides}) != len(overrides):
            raise ValueError("An operator owner cannot have two coordinate overrides.")
        if any(p.target_labels != self.target_labels or p.target_dims != self.target_dims for _, p in overrides):
            raise ValueError("Operator coordinate overrides must use the same target space.")
        object.__setattr__(self, "overrides", overrides)

    @classmethod
    def capture(cls, chip: Any, target_labels: tuple[str, ...], target_dims: tuple[int, ...],
                embedding: Any) -> OperatorProjection:
        """Compose a reduction with the source's already captured operator coordinates."""
        previous = [t.projection for t in chip.effective_terms if t.projection is not None]
        if not previous:
            return cls(tuple(d.label for d in chip.devices), tuple(chip.authored_dims),
                       target_labels, target_dims, embedding)
        if len(previous) != 1 or previous[0].target_labels != tuple(d.label for d in chip.devices):
            raise NotImplementedError("Combining partial operator projections requires a common source space.")
        parent = previous[0]
        def advance(previous: OperatorProjection) -> OperatorProjection:
            return cls(previous.source_labels, previous.source_dims, target_labels, target_dims,
                       previous.embedding @ embedding)

        return replace(advance(parent), overrides=tuple((key, advance(p)) for key, p in parent.overrides))

    def with_current_operators(self, owner_keys: tuple[str, ...]) -> OperatorProjection:
        """Mark operators already authored in this retained space, preserving their lineage."""
        if not owner_keys:
            return self
        current = type(self)(self.target_labels, self.target_dims, self.target_labels,
                             self.target_dims, jnp.eye(prod(self.target_dims), dtype=complex))
        overrides = {**dict(self.overrides), **dict.fromkeys(owner_keys, current)}
        return replace(self, overrides=tuple(overrides.items()))

    def apply(self, operator: Any, labels: tuple[str, ...], owner_key: str | None = None) -> PhysicsExpr:
        """Project a live local array without allocating its full-space identity embedding."""
        for key, projection in self.overrides:
            if key == owner_key:
                return projection.apply(operator, labels)
        support = tuple(self.source_labels.index(label) for label in labels)
        rest = tuple(index for index in range(len(self.source_dims)) if index not in support)
        local_size = prod(self.source_dims[index] for index in support)
        rest_size = prod(self.source_dims[index] for index in rest)
        target_size = prod(self.target_dims)
        matrix = jnp.asarray(operator)
        if matrix.shape != (local_size, local_size):
            raise ValueError("Operator dimensions do not match its captured source support.")
        tensor = jnp.asarray(self.embedding).reshape(self.source_dims + (target_size,))
        tensor = jnp.transpose(tensor, support + rest + (len(self.source_dims),))
        tensor = tensor.reshape(local_size, rest_size, target_size)
        acted = jnp.einsum("ab,brj->arj", matrix, tensor)
        projected = jnp.einsum("ari,arj->ij", tensor.conj(), acted)
        return PhysicsExpr.from_matrix(projected, labels=self.target_labels, dims=self.target_dims,
                                       name="projected_operator")

    def fingerprint(self) -> Any:
        return value_fingerprint((self.source_labels, self.source_dims, self.target_labels,
                                  self.target_dims, self.embedding,
                                  tuple((key, p.fingerprint()) for key, p in self.overrides)))

    def to_dict(self) -> dict[str, Any]:
        matrix = np.asarray(self.embedding)
        return dict(source_labels=list(self.source_labels), source_dims=list(self.source_dims),
                    target_labels=list(self.target_labels), target_dims=list(self.target_dims),
                    real=matrix.real.tolist(), imag=matrix.imag.tolist(),
                    overrides={key: p.to_dict() for key, p in self.overrides})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorProjection:
        if set(data) != {"source_labels", "source_dims", "target_labels", "target_dims", "real", "imag", "overrides"}:
            raise ValueError("Invalid serialized OperatorProjection fields.")
        return cls(tuple(data["source_labels"]), tuple(data["source_dims"]), tuple(data["target_labels"]),
                   tuple(data["target_dims"]), np.asarray(data["real"]) + 1j * np.asarray(data["imag"]),
                   tuple((key, cls.from_dict(p)) for key, p in data["overrides"].items()))


@dataclass(frozen=True, eq=False)
class EffectiveTerms:
    """Retained matrix terms in the named devices' authored coordinates.

    ``hamiltonian`` is in GHz; channels carry unscaled operators and rates in
    1/ns. These terms already express the reduction's selected approximation,
    so assembly changes their frame without dropping further operator bands.
    Values are captured at construction. Editing a surviving device changes
    its authored terms; it does not recompute this captured correction.
    ``projection`` carries the source coordinates of surviving operators;
    channels already stored here are in retained coordinates and bypass it.

    Parameters
    ----------
    labels : tuple[str, ...]
        Device labels defining the retained tensor-product order.
    dims : tuple[int, ...]
        Authored dimensions aligned with ``labels``.
    hamiltonian : array-like
        Hermitian retained Hamiltonian in GHz.
    channels : tuple[CollapseChannel, ...], default=()
        Retained jump operators and Lindblad rates in 1/ns.
    label : str, default="effective"
        Name used for diagnostics and the generated expression.
    projection : OperatorProjection or None, default=None
        Captured source-to-retained operator coordinates.
    """

    labels: tuple[str, ...]
    dims: tuple[int, ...]
    hamiltonian: Any
    channels: tuple[CollapseChannel, ...] = ()
    label: str = "effective"
    projection: OperatorProjection | None = None

    def __post_init__(self) -> None:
        labels, dims = tuple(self.labels), tuple(self.dims)
        if not labels or len(set(labels)) != len(labels) or len(labels) != len(dims):
            raise ValueError("EffectiveTerms requires unique labels and matching dimensions.")
        if any(not isinstance(label, str) or not label for label in labels) or not self.label:
            raise ValueError("EffectiveTerms labels must be nonempty strings.")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in dims):
            raise ValueError("EffectiveTerms dimensions must be positive integers.")
        if self.projection is not None and (
            self.projection.target_labels != labels or self.projection.target_dims != dims
        ):
            raise ValueError("EffectiveTerms projection must use the retained contribution coordinates.")
        shape = (prod(dims), prod(dims))
        h = copy_value(self.hamiltonian, readonly=True)
        if getattr(h, "shape", None) != shape:
            raise ValueError(f"Effective Hamiltonian must have shape {shape}.")
        if not contains_tracer(h):
            concrete = np.asarray(h)
            if not np.all(np.isfinite(concrete)) or not np.allclose(
                concrete, concrete.conj().T, atol=1e-12, rtol=1e-12
            ):
                raise ValueError("Effective Hamiltonian must be finite and Hermitian.")
        if any(not isinstance(channel, CollapseChannel) for channel in self.channels):
            raise TypeError("EffectiveTerms channels must be CollapseChannel values.")
        channels = tuple(
            CollapseChannel(copy_value(c.operator, readonly=True), copy_value(c.rate, readonly=True), c.name)
            for c in self.channels
        )
        if any(getattr(c.operator, "shape", None) != shape for c in channels):
            raise ValueError(f"Effective jump operators must have shape {shape}.")
        for channel in channels:
            if np.shape(channel.rate) != ():
                raise ValueError("Effective channel rates must be scalar.")
            if not contains_tracer((channel.operator, channel.rate)):
                if not np.all(np.isfinite(np.asarray(channel.operator))) or not np.isfinite(channel.rate):
                    raise ValueError("Effective channel operators and rates must be finite.")
        if len({c.name for c in channels}) != len(channels):
            raise ValueError("Effective channel names must be unique within one contribution.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "hamiltonian", h)
        object.__setattr__(self, "channels", channels)

    def expression(self, matrix: Any = None) -> PhysicsExpr:
        """Return a captured matrix through the authored-expression path.

        Parameters
        ----------
        matrix : array-like or None, default=None
            Matrix in GHz. ``None`` uses :attr:`hamiltonian`.
        """
        return PhysicsExpr.from_matrix(
            self.hamiltonian if matrix is None else matrix, labels=self.labels, dims=self.dims, name=self.label
        )

    def validate_for(self, chip: Any) -> None:
        """Validate retained labels and dimensions against a chip.

        Parameters
        ----------
        chip : Chip
            Chip that will receive these terms.
        """
        unknown = set(self.labels) - chip.device_map.keys()
        if unknown:
            raise ValueError(f"Effective terms {self.label!r} target unknown devices {sorted(unknown)}.")
        actual = tuple(chip[label].local_space().dimension for label in self.labels)
        if actual != self.dims:
            raise ValueError(f"Effective terms {self.label!r} require authored dimensions {self.dims}; got {actual}.")

    def fingerprint(self) -> Any:
        return value_fingerprint(
            (
                self.label,
                self.labels,
                self.dims,
                self.hamiltonian,
                tuple((c.name, c.operator, c.rate) for c in self.channels),
                None if self.projection is None else self.projection.fingerprint(),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Persist concrete retained terms without serializing a live calculation."""

        def matrix(value: Any) -> dict[str, Any]:
            array = np.asarray(value, dtype=complex)
            return {"real": array.real.tolist(), "imag": array.imag.tolist()}

        return dict(
            projection=None if self.projection is None else self.projection.to_dict(),
            label=self.label,
            labels=list(self.labels),
            dims=list(self.dims),
            hamiltonian=matrix(self.hamiltonian),
            channels=[dict(name=c.name, operator=matrix(c.operator), rate=float(c.rate)) for c in self.channels],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EffectiveTerms:
        """Recreate validated retained terms from their numerical payload.

        Parameters
        ----------
        data : dict[str, Any]
            Payload produced by :meth:`to_dict`.
        """
        if set(data) != {"label", "labels", "dims", "hamiltonian", "channels", "projection"}:
            raise ValueError("Invalid serialized EffectiveTerms fields.")

        def matrix(value: dict[str, Any]) -> Any:
            return np.asarray(value["real"]) + 1j * np.asarray(value["imag"])

        return cls(
            tuple(data["labels"]),
            tuple(data["dims"]),
            matrix(data["hamiltonian"]),
            tuple(CollapseChannel(matrix(c["operator"]), c["rate"], c["name"]) for c in data["channels"]),
            data["label"],
            None if data["projection"] is None else OperatorProjection.from_dict(data["projection"]),
        )
