"""scqubits interoperability — ``from_scqubits`` / ``to_scqubits`` dispatch.

Importing this subpackage registers the shipped scqubits device mappings (see
:mod:`quchip.interop.scqubits.devices`) with the library-agnostic
:mod:`quchip.interop.base` registry. The two public functions dispatch through
that registry after checking that scqubits is installed.
"""

from __future__ import annotations

from typing import Any

from quchip.interop.base import export_object, import_object

from . import devices  # noqa: F401  (import registers the shipped mappings)


def _require_scqubits() -> None:
    """Import scqubits on demand so quchip remains usable without it."""
    try:
        import scqubits  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError(
            "scqubits is required for this feature. "
            "Install it with:  pip install quchip[scqubits]"
        ) from None


def from_scqubits(obj: Any, **opts: Any) -> Any:
    """Import a scqubits device or composite system.

    Parameters
    ----------
    obj : scqubits object
        A supported circuit, oscillator, or ``HilbertSpace``. Device mappings
        are listed in :mod:`quchip.interop.scqubits.devices`.
    **opts
        Device imports accept ``levels`` (default: source ``truncated_dim``),
        ``label`` (default: source ``id_str``), and target-device noise options
        such as ``T1``, ``T2`` and ``thermal_occupation``. Matrix-element
        relaxation may also require ``coupling_channel``; see the target class.
        Composite imports accept ``frame`` (default ``"lab"``) and
        ``approximation`` (default ``Exact()``); device options are not forwarded.

    Returns
    -------
    BaseDevice or Chip
        Converted device, or a composite of frozen eigenbasis snapshots.
        Snapshot source parameters are not differentiable through quchip.

    Raises
    ------
    ImportError
        The optional ``quchip[scqubits]`` dependency is unavailable.
    LookupError
        No mapping exists for the source type.
    NotImplementedError
        A composite contains unsupported interactions.

    References
    ----------
    Groszkowski and Koch, *scqubits: a Python package for superconducting
    qubits*, Quantum 5, 583 (2021), https://doi.org/10.22331/q-2021-11-17-583.
    """
    _require_scqubits()

    import scqubits

    if isinstance(obj, scqubits.HilbertSpace):
        from .composite import import_hilbertspace

        return import_hilbertspace(obj, **opts)

    return import_object(obj, **opts)


def to_scqubits(device_or_chip: Any, **opts: Any) -> Any:
    """Export a quchip device or chip to scqubits.

    Parameters
    ----------
    device_or_chip : BaseDevice or Chip
        Model with concrete parameters. A chip exports subsystems and supported
        interactions; filtered couplings require ``Exact()`` before export.
    **opts
        Mapping-specific options. ``DuffingTransmon`` export accepts ``ncut``
        (integer charge cutoff, default 30) for reconstructing circuit energies.
        Other shipped device mappings currently ignore extra options.
        Composite export accepts no keyword options.

    Returns
    -------
    scqubits object
        Corresponding device or ``HilbertSpace``. Chip control equipment and
        baths are omitted with a warning; port networks and effective terms
        are unsupported. See :func:`~quchip.interop.scqubits.composite.export_chip`.

    Raises
    ------
    ImportError
        The optional ``quchip[scqubits]`` dependency is unavailable.
    LookupError
        No export mapping exists for a device type.
    ValueError
        Parameters are traced or approximation filtering would change the export.

    References
    ----------
    Groszkowski and Koch, Quantum 5, 583 (2021),
    https://doi.org/10.22331/q-2021-11-17-583.
    """
    _require_scqubits()

    from quchip.chip.chip import Chip

    if isinstance(device_or_chip, Chip):
        from .composite import export_chip

        return export_chip(device_or_chip, **opts)

    return export_object(device_or_chip, "scqubits", **opts)


__all__ = ["from_scqubits", "to_scqubits"]
