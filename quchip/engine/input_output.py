"""Backend-neutral input-output assembly for stationary port calculations.

Accessible channels are the port-marked subset of
:class:`~quchip.engine.ir.CollapseTerm`. This module derives coherent input Hamiltonians, output coupling
operators, and stationary-frame checks from those resolved terms. Numerical
Liouvillian construction and solution remain backend-owned.
"""

from __future__ import annotations

from typing import Any

from quchip.chip.partition import connected_components
from quchip.engine.ir import CanonicalOperator, EngineResult, StaticTerm
from quchip.engine.reference import cw_transfer
from quchip.utils.jax_utils import maybe_concrete_scalar


def resolve_stationary_engine(
    chip: Any,
    port_frequencies: tuple[tuple[str, Any], ...],
) -> EngineResult:
    """Resolve port-tone frequencies into one static engine description."""
    if not port_frequencies:
        raise ValueError("Stationary port analysis requires at least one tone frequency.")

    reference = port_frequencies[0][1]
    one_carrier = all(
        same_frequency(reference, frequency)
        for _, frequency in port_frequencies[1:]
    )
    if one_carrier:
        engine = chip.resolve(frame=reference)
        _validate_port_frequencies(engine, port_frequencies)
        if engine.dynamic_terms:
            raise ValueError(
                "The selected tones leave dynamic Hamiltonian terms after frame and approximation resolution. "
                "Use QuantumSequence for time evolution; periodic/Floquet steady states are not supported."
            )
        return engine

    device_frequencies: dict[str, Any] = {}
    for port_label, frequency in port_frequencies:
        targets = _tone_targets(chip, port_label)
        for target in targets:
            assigned = device_frequencies.get(target)
            if assigned is not None and not same_frequency(assigned, frequency):
                raise ValueError(
                    f"Ports address {target!r} with distinct stationary tones. "
                    "Use QuantumSequence for time evolution; periodic/Floquet steady states are not supported."
                )
            device_frequencies[target] = frequency

    exchange_edges = (
        (coupling.device_a_label, coupling.device_b_label)
        for coupling in chip.couplings
        if getattr(coupling, "folds_exchange", False)
    )
    labels = tuple(device.label for device in chip.devices)
    for component in connected_components(labels, exchange_edges):
        assigned = [device_frequencies[label] for label in component if label in device_frequencies]
        if not assigned:
            continue
        reference = assigned[0]
        if any(not same_frequency(reference, frequency) for frequency in assigned[1:]):
            raise ValueError(
                f"Exchange-connected devices {component!r} are addressed by distinct stationary tones. "
                "Use QuantumSequence for time evolution; periodic/Floquet steady states are not supported."
            )
        device_frequencies.update({label: reference for label in component})

    engine = chip.resolve(frame=device_frequencies)
    if engine.dynamic_terms:
        raise ValueError(
            "The selected tones leave dynamic Hamiltonian terms after frame and approximation resolution. "
            "Use QuantumSequence for time evolution; periodic/Floquet steady states are not supported."
        )
    _validate_port_frequencies(engine, port_frequencies)
    return engine


def _tone_targets(chip: Any, exposure: str) -> tuple[str, ...]:
    """Return the devices whose ports a tone entering ``exposure`` structurally reaches."""
    from quchip.chip.port_network import PortNetwork

    network = chip.port_network
    if network is None:
        return chip.port(exposure).resolve_targets(chip)
    exposures, _, coupling_maps, _, _, support = network._compile()
    labels = [item.label for item in exposures]
    if exposure not in labels:
        raise ValueError(f"Unknown resolved port {exposure!r}.")
    column = labels.index(exposure)
    targets: list[str] = []
    for row, mapping in enumerate(coupling_maps):
        if not support[row, column]:
            continue
        for source in PortNetwork._active_mapping_sources(mapping):
            targets.extend(chip.port(source).resolve_targets(chip))
    return tuple(dict.fromkeys(targets))


def port_operators(engine: EngineResult, backend: Any) -> dict[str, CanonicalOperator]:
    """Return each resolved channel coupling ``L_p = exp(i phi) sqrt(kappa) A_p``."""
    operators: dict[str, CanonicalOperator] = {}
    for channel in engine.slh.channels:
        operators[channel.key] = channel.coupling.with_metadata(tag=f"port:{channel.key}")
    return operators


def add_port_inputs(
    engine: EngineResult,
    backend: Any,
    tones: tuple[tuple[str, Any, Any], ...],
) -> EngineResult:
    """Add stationary ``i(beta* L - beta L†)`` terms in angular units."""
    if not tones:
        return engine
    xp = backend.array_module
    external = engine.slh.external_channels
    operators = port_operators(engine, backend)
    exposure_index = {channel.key: index for index, channel in enumerate(external)}
    _validate_port_frequencies(
        engine,
        tuple((port_label, frequency) for port_label, frequency, _ in tones),
    )
    incident = [xp.asarray(0.0 + 0.0j) for _ in external]
    for port_label, frequency, amplitude in tones:
        input_index = exposure_index[port_label]
        incident[input_index] = incident[input_index] + xp.asarray(amplitude) * cw_transfer(
            external[input_index].reference.inbound, frequency, xp
        )

    terms = list(engine.applied_hamiltonian.static_terms)
    for output_index, channel in enumerate(engine.slh.channels):
        coefficient = xp.asarray(0.0 + 0.0j)
        for input_index, amplitude in enumerate(incident):
            coefficient = coefficient + xp.asarray(
                engine.slh.S[output_index, input_index]
            ) * amplitude
        coupling = operators[channel.key]
        values = coupling.to_dense()
        h_input = 1j * (
            xp.conj(coefficient) * values
            - coefficient * xp.conj(xp.swapaxes(values, -1, -2))
        )
        terms.append(
            StaticTerm(
                CanonicalOperator.from_dense(
                    h_input,
                    dims=coupling.dims,
                    basis=coupling.basis,
                    subsystem_labels=coupling.subsystem_labels,
                    tag=f"input:{channel.key}",
                ),
                origin="port",
                metadata={"port": channel.key},
            )
        )
    return engine.with_applied_hamiltonian_terms(static_terms=tuple(terms))


def _validate_port_frequencies(
    engine: EngineResult,
    port_frequencies: tuple[tuple[str, Any], ...],
) -> None:
    """Check that every channel a tone feeds is stationary in the tone's frame."""
    external = {channel.key: index for index, channel in enumerate(engine.slh.external_channels)}
    for port_label, frequency in port_frequencies:
        if port_label not in external:
            raise ValueError(f"Unknown resolved port {port_label!r}.")
        column = external[port_label]
        for row, channel in enumerate(engine.slh.channels):
            resolved = channel.collapse.frame_frequency
            if resolved is None or not engine.slh.feeds(row, column):
                continue
            if not same_frequency(resolved, frequency):
                raise ValueError(
                    f"Tone at {frequency!r} GHz entering {port_label!r} reaches channel "
                    f"{channel.key!r}, which is stationary at {resolved!r} GHz. "
                    "Use QuantumSequence for time evolution."
                )


def same_frequency(first: Any, second: Any) -> bool:
    """Compare concrete frequencies without forcing traced values to Python."""
    if first is second:
        return True
    first_value = maybe_concrete_scalar(first)
    second_value = maybe_concrete_scalar(second)
    return (
        first_value is not None
        and second_value is not None
        and first_value == second_value
    )
