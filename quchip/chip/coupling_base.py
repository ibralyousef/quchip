"""Coupling base class and registry.

Defined separately from concrete couplings to allow declarative models to
subclass :class:`BaseCoupling` without a circular import.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from quchip.backend.protocol import Operator
from quchip.declarative.dissipation import CollapseChannel
from quchip.devices.base import BaseDevice
from quchip.utils.labeling import auto_label, resolve_label
from quchip.utils.registry import Registrable
from quchip.utils.state_versioning import StateVersioned

if TYPE_CHECKING:
    from quchip.engine.ir import DroppedTerm


class BaseCoupling(StateVersioned, Registrable, ABC, registry_root=True):
    """Abstract base for a two-body coupling between :class:`BaseDevice`s.

    Subclasses own their local interaction Hamiltonian ``H_int`` acting on
    ``H_a ⊗ H_b``. The engine embeds this local form into the full chip
    tensor space; couplings never touch the engine directly.

    Subclasses auto-register via the shared
    :class:`~quchip.utils.registry.Registrable` mixin — no manual
    registration step is needed. Ensure the module defining the subclass is
    imported so the registration runs.

    Parameters
    ----------
    device_a, device_b : BaseDevice or str
        The two coupled devices, given as device objects or their label
        strings. Label-string references are *late-bound*: the coupling
        remembers the string, and :class:`Chip`
        resolves it to the matching device instance at construction time.
        Before that resolution the coupling cannot produce an interaction
        Hamiltonian.
    label : str, optional
        Human-readable label. Auto-generated from ``_type_prefix`` when
        omitted (e.g. ``"cap_0"`` for :class:`Capacitive`).

    Notes
    -----
    All coupling parameters (``g``, any tunable envelope, etc.) must be
    JAX-traceable so sweeps and gradient-based optimization work without
    forcing concretization.
    """

    _type_prefix: ClassVar[str] = "coupling"


    # Whether dressed-specification fitting estimates this exchange strength
    # from a dispersive pull, rather than binding the pull as a direct scalar.
    reduces_to_crosskerr: ClassVar[bool] = False

    #: Observable represented by this coupling's declared scalar when the
    #: enclosing chip is passed to ``fit_a_dress`` as a dressed
    #: specification. Concrete coupling models override this when their
    #: inverse-design quantity is a dressed interaction observable.
    default_fit_observable: ClassVar[str] = "coupling_strength"

    # The endpoint device references are structural — a rebinding during
    # ``copy()`` is not a physics change — so they must not bump state_version.
    # Mutation tracking, the seed, ``state_version`` and ``_finish_init`` are
    # owned by the shared StateVersioned mixin; tracking switches on
    # automatically once construction finishes (no manual ``_finish_init``).
    _untracked_names = frozenset({"device_a", "device_b"})

    def __init__(
        self,
        device_a: BaseDevice | str,
        device_b: BaseDevice | str,
        *,
        label: str | None = None,
    ) -> None:
        """Store endpoint references and assign a stable coupling label."""
        for name, dev in (("device_a", device_a), ("device_b", device_b)):
            if not isinstance(dev, (BaseDevice, str)):
                raise TypeError(
                    f"{name} must be a BaseDevice or label string, got {type(dev).__name__}."
                )
        self.device_a = device_a
        self.device_b = device_b
        self.label = label if label is not None else auto_label(type(self)._type_prefix)

    def copy(self, device_map: dict[str, BaseDevice]) -> "BaseCoupling":
        """Copy authored values and rebind endpoints to *device_map*."""
        from quchip.declarative.parameters import copy_authored_fields

        cloned = copy.copy(self)
        copy_authored_fields(self, cloned)
        object.__setattr__(cloned, "device_a", device_map[self.device_a_label])
        object.__setattr__(cloned, "device_b", device_map[self.device_b_label])
        return cloned

    def parameter_values(self) -> dict[str, Any]:
        """Return this coupling's bindable values by local field name."""
        fields = getattr(type(self), "__quchip_param_fields__", None)
        if fields is not None:
            return {name: getattr(self, name) for name in fields}
        return {self.coupling_strength_name: self.coupling_strength}

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Apply one local parameter value on an isolated coupling copy."""
        fields = getattr(type(self), "__quchip_param_fields__", None)
        if fields is not None and name in fields:
            setattr(self, name, value)
            return
        if fields is None and name == self.coupling_strength_name:
            self.set_coupling_strength(value)
            return
        raise KeyError(name)

    @property
    def device_a_label(self) -> str:
        """Label of the first coupled device (works pre- and post-binding)."""
        return resolve_label(self.device_a)

    @property
    def device_b_label(self) -> str:
        """Label of the second coupled device (works pre- and post-binding)."""
        return resolve_label(self.device_b)

    @property
    def is_resolved(self) -> bool:
        """Whether both device references are bound to :class:`BaseDevice` instances."""
        return isinstance(self.device_a, BaseDevice) and isinstance(self.device_b, BaseDevice)

    def __repr__(self) -> str:
        """Return a minimal endpoint summary (default for full-control subclasses)."""
        return f"{type(self).__name__}('{self.device_a_label}' <-> '{self.device_b_label}', label={self.label!r})"

    def _resolve_devices(self, device_map: dict[str, BaseDevice]) -> None:
        """Validate both endpoints before binding label references to chip devices."""
        endpoints = {}
        for attr in ("device_a", "device_b"):
            current = getattr(self, attr)
            label = resolve_label(current)
            resolved = device_map.get(label)
            if resolved is None:
                raise ValueError(
                    f"Coupling {self!r} references device {label!r} which is not in the device list. "
                    f"Available labels: {list(device_map)}"
                )
            if not isinstance(current, str) and current is not resolved:
                raise ValueError(
                    f"Coupling {self.label!r} references a different device object labeled {label!r}. "
                    "Use the declared chip device or its label."
                )
            endpoints[attr] = resolved
        for attr, resolved in endpoints.items():
            object.__setattr__(self, attr, resolved)

    @property
    @abstractmethod
    def coupling_strength(self) -> float:
        """Scalar coupling strength in GHz."""
        ...

    @property
    def coupling_strength_name(self) -> str:
        """Display name of the scalar :attr:`coupling_strength` parameter.

        Default ``"g"`` (the conventional coupling-strength symbol, and the
        actual attribute name on :class:`~quchip.chip.couplings.Coupling`).
        :class:`~quchip.declarative.models.CouplingModel` overrides this to
        the name of its first declared parameter field; a subclass with a
        different primary-scalar convention overrides it directly.
        """
        return "g"

    def set_coupling_strength(self, value: Any) -> None:
        """Write the primary scalar named by :attr:`coupling_strength_name`.

        Examples include ``g``, ``g_0``, and ``chi``. Override when the writable
        attribute differs from that name.
        """
        setattr(self, self.coupling_strength_name, value)

    def default_dressed_target(self) -> tuple[str, Any]:
        """Return this edge's component-owned inverse-design constraint.

        The value is read directly from the declared coupling scalar.  No
        chip-level observable is evaluated here.
        """
        return self.default_fit_observable, self.coupling_strength

    @abstractmethod
    def interaction_hamiltonian(self) -> Operator:
        """Return the full ``H_int`` on the local ``H_a ⊗ H_b`` subspace.

        Couplings author one complete interaction. The selected engine
        approximation decides which resolved bands are retained.
        """
        ...

    def physics_notes(self) -> list[str]:
        """Return human-readable declarations of this coupling's approximations."""
        return [
            f"Coupled devices: '{self.device_a_label}' ↔ '{self.device_b_label}'",
        ]

    def dropped_terms(self) -> list["DroppedTerm"]:
        """Return advisory records for terms this coupling's model itself elides.

        RWA band drops are reported generically during assembly; this hook is
        for *other* approximations a coupling applies inside
        :meth:`interaction_hamiltonian`. Default: nothing is dropped.
        """
        return []

    def _time_terms(self) -> tuple[Any, ...]:
        """Return normalized time-dependent terms (default: none)."""
        return ()

    def collapse_channels(self) -> tuple[CollapseChannel, ...]:
        """Local Lindblad channels contributed by this coupling."""
        return ()

    def _collapse_channels_with_paths(
        self,
    ) -> tuple[tuple[CollapseChannel, tuple[str, ...]], ...]:
        """Return normalized channels with inferred paths (none for legacy couplings)."""
        return ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize structural fields into a JSON-safe dictionary."""
        data = super().to_dict()
        data["device_a_label"] = self.device_a_label
        data["device_b_label"] = self.device_b_label
        data["label"] = self.label
        return data
