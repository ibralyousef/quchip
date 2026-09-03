"""Composable SLH field boundaries and their resolved physics."""

from __future__ import annotations

from typing import Any

import json

import numpy as np
import pytest

from quchip import Chip, Port, PortNetwork, QuantumSequence, Resonator


def _lowering(levels: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, levels)), 1).astype(complex)


def test_network_owns_ports_and_unconnected_ports_are_identity_exposures() -> None:
    """Unconnected quantum ports resolve as identity external exposures."""
    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(label="feedline")
    left = network.port("left", target=resonator, rate=0.01)
    right = network.port("right", target=resonator, rate=0.04)

    chip = Chip([resonator], port_network=network)
    resolved = chip.resolve().slh

    assert chip.port_network is network
    assert chip.ports == (left, right)
    assert [channel.key for channel in resolved.external_channels] == ["left", "right"]
    np.testing.assert_allclose(resolved.S, np.eye(2))
    np.testing.assert_allclose(resolved.L[0].to_dense(), np.sqrt(0.01) * _lowering(3))
    np.testing.assert_allclose(resolved.L[1].to_dense(), np.sqrt(0.04) * _lowering(3))


def test_direct_scattering_mapping_uses_output_input_order() -> None:
    """Scattering mappings use the documented output-input key order."""
    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(
        label="feedline",
        scattering={("right", "left"): 1.0, ("left", "right"): 1.0},
    )
    network.port("left", target=resonator, rate=0.01)
    network.port("right", target=resonator, rate=0.04)

    resolved = Chip([resonator], port_network=network).resolve().slh

    np.testing.assert_allclose(network.S, [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(resolved.S, [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), np.sqrt(0.04) * _lowering(3))
    np.testing.assert_allclose(resolved.L[1].to_dense(), np.sqrt(0.01) * _lowering(3))


def test_concrete_nonunitary_scattering_is_rejected() -> None:
    """A concrete scalar scattering boundary must be unitary."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(scattering=[[0.9]], label="lossy")
    network.port("readout", target=resonator, rate=0.01)

    with pytest.raises(ValueError, match="unitary"):
        Chip([resonator], port_network=network).resolve()


def test_cascade_requires_an_explicit_remaining_boundary() -> None:
    """A cascade hides connected terminals until the remaining boundary is exposed."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)

    with pytest.raises(ValueError, match="free terminals"):
        Chip([first, second], port_network=network).resolve()


def test_instantaneous_feedback_cycle_is_rejected_explicitly() -> None:
    """Instantaneous network feedback cycles fail before model resolution."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="loop")
    port = network.port("chip_port", target=resonator, rate=0.04)
    line = network.through("line")
    network.connect(port.output, line.input)
    network.connect(line.output, port.input)

    with pytest.raises(ValueError, match="feedback|cycle"):
        Chip([resonator], port_network=network).resolve()


def test_cascade_generates_series_coupling_and_hamiltonian() -> None:
    """Series composition produces both combined coupling and an SLH Hamiltonian."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)
    network.expose("feedline", input=a.input, output=b.output)

    resolved = Chip([first, second], port_network=network).resolve().slh
    identity = np.eye(2)
    l_a = np.sqrt(0.04) * np.kron(_lowering(2), identity)
    l_b = np.sqrt(0.09) * np.kron(identity, _lowering(2))
    product = l_b.conj().T @ l_a
    expected_h = (product - product.conj().T) / (2j)

    assert [channel.key for channel in resolved.external_channels] == ["feedline"]
    np.testing.assert_allclose(resolved.S, [[1.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), l_a + l_b)
    generated = [
        term for term in resolved.H.static_terms if term.origin == "network"
    ]
    assert len(generated) == 1
    np.testing.assert_allclose(generated[0].operator.to_dense(), expected_h)


def test_cascade_rejects_mixed_rotating_frame_frequencies() -> None:
    """Static SLH composition refuses channels with a missing relative carrier."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)
    network.expose("feedline", input=a.input, output=b.output)
    chip = Chip(
        [first, second],
        port_network=network,
        frame={"a": 5.0, "b": 6.0},
    )

    with pytest.raises(ValueError, match="different rotating-frame frequencies"):
        chip.resolve()


def test_phase_component_enters_series_coupling_and_generated_hamiltonian() -> None:
    """A phase shifter rotates both series coupling and its generated Hamiltonian."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    phase = network.phase_shift("phase", phase=np.pi / 2)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, phase, b)
    network.expose("feedline", input=a, output=b)

    resolved = Chip([first, second], port_network=network).resolve().slh
    identity = np.eye(2)
    l_a = np.sqrt(0.04) * np.kron(_lowering(2), identity)
    l_b = np.sqrt(0.09) * np.kron(identity, _lowering(2))
    propagated = 1j * l_a
    product = l_b.conj().T @ propagated

    np.testing.assert_allclose(resolved.S, [[1j]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), propagated + l_b)
    generated = [term for term in resolved.H.static_terms if term.origin == "network"]
    np.testing.assert_allclose(
        generated[0].operator.to_dense(),
        (product - product.conj().T) / (2j),
    )


def test_cascade_and_expose_accept_terminals_ports_and_components() -> None:
    """One cascade call chains mixed endpoints; ambiguous shorthand fails loudly."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    line = network.phase_shift("line", phase=0.0)
    splitter = network.beam_splitter("splitter")
    network.cascade(port, line, splitter.input_terminal("left"))
    network.expose("readout", input=port, output=splitter.output_terminal("left"))

    with pytest.raises(AttributeError, match="multiple inputs"):
        network.expose("other", input=splitter, output=splitter.output_terminal("right"))
    with pytest.raises(ValueError, match="output terminal"):
        network.cascade(splitter.input_terminal("right"), port)

    network.expose("spare", input=splitter.input_terminal("right"), output=splitter.output_terminal("right"))

    resolved = Chip([resonator], port_network=network).resolve().slh
    transmitted = np.sqrt(0.5) * np.sqrt(0.04) * _lowering(2)
    np.testing.assert_allclose(resolved.L[0].to_dense(), transmitted)
    np.testing.assert_allclose(resolved.L[1].to_dense(), -transmitted)


def test_sequence_template_retains_composed_input_free_slh() -> None:
    """Sequence assembly preserves the resolved input-free network model."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)
    network.expose("feedline", input=a.input, output=b.output)
    chip = Chip([first, second], port_network=network)

    resolved = QuantumSequence(chip).resolve().slh

    assert [channel.key for channel in resolved.external_channels] == ["feedline"]
    np.testing.assert_allclose(resolved.S, [[1.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), chip.resolve().slh.L[0].to_dense())


def test_delay_section_is_reference_plane_metadata_only() -> None:
    """A linked delay decorates both legs of the plane without entering instantaneous SLH."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.01)
    cable = network.delay("cable", duration=0.25)
    network.link(port, cable)
    network.expose("readout", at=cable.side(2))

    resolved = Chip([resonator], port_network=network).resolve().slh
    plane = resolved.external_channels[0].reference

    assert [element.duration for element in plane.inbound] == [0.25]
    assert [element.duration for element in plane.outbound] == [0.25]
    np.testing.assert_allclose(resolved.S, [[1.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), np.sqrt(0.01) * _lowering(2))


def test_delay_section_after_a_circulator_decorates_only_the_reached_legs() -> None:
    """Reference sections apply per propagation path; interior sections are rejected."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="fridge")
    port = network.port("chip_port", target=resonator, rate=0.04)
    circulator = network.circulator("circ")
    line = network.delay("line", duration=1.5)
    network.link(port, circulator.side(2))
    network.link(circulator.side(3), line)
    network.expose("drive", at=circulator.side(1))
    network.expose("readout", at=line.side(2))
    chip = Chip([resonator], port_network=network)

    resolved = chip.resolve().slh
    drive_plane, readout_plane = (channel.reference for channel in resolved.external_channels)

    assert drive_plane.inbound == () and drive_plane.outbound == ()
    assert [element.label for element in readout_plane.inbound] == ["line"]
    assert [element.label for element in readout_plane.outbound] == ["line"]
    rebound = chip.with_params({"network.component.line.duration": 2.5}).resolve().slh
    assert rebound.external_channels[1].reference.outbound[0].duration == 2.5


def test_reference_section_between_core_components_is_rejected() -> None:
    """A delay must sit between an exposure and the Markov boundary, never inside the core."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    interior = PortNetwork(label="interior")
    inner_port = interior.port("chip_port", target=resonator, rate=0.04)
    inner_loss = interior.attenuator("loss", eta=0.5)
    inner_cable = interior.delay("cable", duration=0.1)
    interior.link(inner_port, inner_cable, inner_loss)
    interior.expose("readout", at=inner_loss.side(2))
    with pytest.raises(ValueError, match="Reference components"):
        Chip([resonator], port_network=interior).resolve()


def test_attenuator_is_a_reciprocal_two_sided_vacuum_dilation() -> None:
    """A linked attenuator attenuates both directions through hidden vacuum channels."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.link(port, loss)
    network.expose("readout", at=loss.side(2))

    resolved = Chip([resonator], port_network=network).resolve().slh
    coupling = np.sqrt(0.04) * _lowering(2)

    assert [channel.key for channel in resolved.channels] == [
        "readout",
        "hidden.cold_loss.vacuum_1",
        "hidden.cold_loss.vacuum_2",
    ]
    assert [channel.accessibility for channel in resolved.channels] == ["exposed", "hidden", "hidden"]
    np.testing.assert_allclose(
        resolved.S,
        [[0.64, 0.6, 0.48], [-0.48, 0.8, -0.36], [-0.6, 0.0, 0.8]],
    )
    np.testing.assert_allclose(resolved.L[0].to_dense(), 0.8 * coupling)
    np.testing.assert_allclose(resolved.L[1].to_dense(), -0.6 * coupling)
    np.testing.assert_allclose(resolved.L[2].to_dense(), 0.0 * coupling)
    dissipative_strength = sum(
        operator.to_dense().conj().T @ operator.to_dense() for operator in resolved.L
    )
    np.testing.assert_allclose(dissipative_strength, coupling.conj().T @ coupling)


def test_network_dilation_precedes_stable_identity_hidden_baths() -> None:
    """Network vacuum channels precede identity scattering for device baths."""
    resonator = Resonator(
        freq=6.0,
        levels=2,
        internal_quality_factor=100_000,
        label="r",
    )
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.link(port, loss)
    network.expose("readout", at=loss.side(2))

    resolved = Chip([resonator], port_network=network).resolve().slh

    assert [channel.key for channel in resolved.channels[:3]] == [
        "readout",
        "hidden.cold_loss.vacuum_1",
        "hidden.cold_loss.vacuum_2",
    ]
    assert resolved.channels[3].collapse.source == "r"
    np.testing.assert_allclose(resolved.S[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(resolved.S[:3, 3], 0.0)


def test_from_ports_builds_an_explicit_identity_network() -> None:
    """Existing port objects can seed the sole explicit network boundary."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    port = Port(resonator, rate=0.01, label="readout")

    chip = Chip([resonator], port_network=PortNetwork.from_ports([port]))

    assert chip.port_network is not None
    assert chip.port_network.ports == (port,)
    np.testing.assert_allclose(chip.resolve().slh.S, [[1.0]])


def test_chip_has_one_network_construction_path() -> None:
    """Ports enter a chip through its sole network-owned boundary."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    port = Port(resonator, rate=0.01, label="readout")

    with pytest.raises(TypeError, match="ports"):
        Chip([resonator], ports=[port])  # type: ignore[call-arg]


def test_second_network_cannot_silently_replace_the_first() -> None:
    """A chip rejects replacing an already attached field network."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    first = PortNetwork(label="first")
    first.port("readout", target=resonator, rate=0.01)
    second = PortNetwork(label="second")
    second.port("other", target=resonator, rate=0.01)
    chip = Chip([resonator], port_network=first)

    with pytest.raises(ValueError, match="disconnect_network"):
        chip.connect_network(second)

    assert chip.disconnect_network() is first
    assert chip.port_network is None
    assert chip.ports == ()


def test_network_graph_round_trips_and_clone_remains_independent() -> None:
    """Serialization preserves the graph while cloning isolates mutable structure."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.link(port, loss)
    cable = network.delay("cable", duration=0.1)
    network.link(loss, cable)
    network.expose("readout", at=cable.side(2))
    chip = Chip([resonator], port_network=network)

    restored = Chip.from_dict(json.loads(json.dumps(chip.to_dict())))
    cloned = chip.clone()

    np.testing.assert_allclose(restored.resolve().slh.S, chip.resolve().slh.S)
    np.testing.assert_allclose(restored.resolve().slh.L[0].to_dense(), chip.resolve().slh.L[0].to_dense())
    assert restored.resolve().slh.external_channels[0].reference == chip.resolve().slh.external_channels[0].reference
    assert cloned.port_network is not chip.port_network
    assert cloned.ports[0] is not chip.ports[0]
    cloned.ports[0].rate = 0.09
    assert chip.ports[0].rate == 0.04
    rebound = chip.with_params({"network.component.cold_loss.eta": 0.25})
    np.testing.assert_allclose(rebound.resolve().slh.S[0, :3], [0.25, np.sqrt(0.75), 0.5 * np.sqrt(0.75)])
    np.testing.assert_allclose(chip.resolve().slh.S[0, :3], [0.64, 0.6, 0.48])


def test_scattering_entries_are_bindable_network_parameters() -> None:
    """Scattering entries rebind through stable parameter paths."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(
        label="line",
        scattering={("readout", "readout"): 1.0},
    )
    network.port("chip_port", target=resonator, rate=0.04)
    network.expose("readout", at=network.ports[0])
    chip = Chip([resonator], port_network=network)

    rebound = chip.with_params({"network.scattering.readout.readout": -1.0})

    np.testing.assert_allclose(rebound.resolve().slh.S, [[-1.0]])
    np.testing.assert_allclose(chip.resolve().slh.S, [[1.0]])


def test_attenuator_power_transmission_is_jax_differentiable() -> None:
    """Attenuator transmission remains differentiable through network resolution."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    resonator = Resonator(freq=6.0, levels=2, label="r")

    def transmitted_coupling(eta):
        network = PortNetwork(label="line")
        port = network.port("chip_port", target=resonator, rate=0.04)
        loss = network.attenuator("cold_loss", eta=eta)
        network.link(port, loss)
        network.expose("readout", at=loss.side(2))
        value = Chip([resonator], port_network=network).resolve().slh.L[0].to_dense()[0, 1]
        return jnp.real(value)

    np.testing.assert_allclose(transmitted_coupling(0.64), 0.16)
    np.testing.assert_allclose(jax.grad(transmitted_coupling)(0.64), 0.125)


def test_circulator_and_isolator_route_a_reflection_readout() -> None:
    """Linked sides wire both directions; permutation rows compile terminal by terminal."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="fridge")
    port = network.port("chip_port", target=resonator, rate=0.04)
    circulator = network.circulator("circ")
    isolator = network.isolator("iso")
    network.link(port, circulator.side(2))
    network.link(circulator.side(3), isolator)
    network.expose("drive", at=circulator.side(1))
    network.expose("readout", at=isolator.side(2))
    chip = Chip([resonator], port_network=network)

    with pytest.raises(ValueError, match="side"):
        network.expose("ambiguous", at=isolator)

    resolved = chip.resolve().slh
    coupling = np.sqrt(0.04) * _lowering(2)

    assert [channel.key for channel in resolved.channels] == ["drive", "readout", "hidden.iso.load"]
    np.testing.assert_allclose(resolved.S, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), 0.0 * coupling)
    np.testing.assert_allclose(resolved.L[1].to_dense(), coupling)
    assert not [term for term in resolved.H.static_terms if term.origin == "network"]

    restored = Chip.from_dict(json.loads(json.dumps(chip.to_dict())))
    np.testing.assert_allclose(restored.resolve().slh.S, resolved.S)
    np.testing.assert_allclose(chip.clone().resolve().slh.L[1].to_dense(), coupling)


def test_link_rejects_directional_components_and_keeps_true_feedback_explicit() -> None:
    """Only sided components link; a bidirectional line between two ports is feedback."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    directional = PortNetwork(label="directional")
    with pytest.raises(ValueError, match="side"):
        directional.link(
            directional.port("a_port", target=first, rate=0.04),
            directional.phase_shift("line", phase=0.1),
        )

    network = PortNetwork(label="loop")
    network.link(
        network.port("a_port", target=first, rate=0.04),
        network.port("b_port", target=second, rate=0.09),
    )
    with pytest.raises(ValueError, match="feedback|cycle"):
        Chip([first, second], port_network=network).resolve()


def test_every_builtin_component_round_trips_as_kind_and_parameters() -> None:
    """Serialized networks rebuild built-in components from their factory kind and parameters."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="fridge")
    port = network.port("chip_port", target=resonator, rate=0.04)
    circ = network.circulator("circ", ports=4)
    iso = network.isolator("iso")
    loss = network.attenuator("loss", eta=0.25)
    splitter = network.beam_splitter("split", eta=0.3)
    hybrid = network.hybrid90("hyb")
    swap = network.permutation("swap", order=[1, 0])
    phase = network.phase_shift("phase", phase=0.4)
    through = network.through("thru")
    custom = network.component("custom", scattering=[[0.0, 1j], [1j, 0.0]], terminals=("a", "b"))
    cable = network.delay("cable", duration=1.5)
    hemt = network.amplifier("hemt", gain=50.0, added_noise=1.0)
    network.link(port, circ.side(2))
    network.link(circ.side(3), iso, loss, cable, hemt)
    network.expose("readout", at=hemt.side(2))
    network.expose("drive", at=circ.side(1))
    network.expose("spare", at=circ.side(4))
    network.cascade(splitter.output_terminal("left"), phase, through, hybrid.input_terminal("left"))
    network.cascade(splitter.output_terminal("right"), swap.input_terminal("0"))
    network.cascade(hybrid.output_terminal("left"), custom.input_terminal("a"))
    network.cascade(swap.output_terminal("0"), custom.input_terminal("b"))
    network.expose("aux_a", input=splitter.input_terminal("left"), output=custom.output_terminal("a"))
    network.expose("aux_b", input=splitter.input_terminal("right"), output=custom.output_terminal("b"))
    network.expose("hyb_side", input=hybrid.input_terminal("right"), output=hybrid.output_terminal("right"))
    network.expose("swap_side", input=swap.input_terminal("1"), output=swap.output_terminal("1"))
    chip = Chip([resonator], port_network=network)

    payload = json.loads(json.dumps(chip.to_dict()))
    restored = Chip.from_dict(payload)

    kinds = {item["label"]: item for item in payload["port_network"]["components"]}
    assert kinds["circ"] == {"label": "circ", "kind": "circulator", "parameters": {"ports": 4}}
    assert kinds["swap"]["parameters"] == {"order": [1, 0]}
    assert kinds["custom"]["kind"] == "scattering" and kinds["custom"]["terminals"] == ["a", "b"]
    assert restored.port_network is not None
    assert restored.port_network.to_dict() == chip.port_network.to_dict()
    original, rebuilt = chip.resolve().slh, restored.resolve().slh
    np.testing.assert_allclose(rebuilt.S, original.S)
    assert [c.key for c in rebuilt.channels] == [c.key for c in original.channels]
    assert restored.parameters == chip.parameters
    assert "component.circ.ports" not in network.parameters
    assert "component.swap.order" not in network.parameters
    with pytest.raises(TypeError, match="kind"):
        PortNetwork.from_dict({"components": [{"label": "x", "kind": "warp_drive", "parameters": {}}]})
    with pytest.raises(TypeError, match="no kind"):
        PortNetwork.from_dict({"components": [{"label": "x", "parameters": {}}]})


def _open_system(network: PortNetwork, chip: Chip) -> tuple[Any, dict[str, str]]:
    """Every core terminal as one channel of an unconnected SLH triple (ports carry L, components none).

    ``network`` must have no connections so every port resolves to its own exposed channel.
    """
    from quchip.engine import concatenate
    from quchip.engine.ir import CollapseTerm, HamiltonianProgram, ResolvedSLH, SLHChannel

    resolved_ports = {channel.key: channel for channel in chip.resolve().slh.external_channels}
    zero = next(iter(resolved_ports.values())).coupling.scaled(0.0)
    systems = []
    for component in network.components:
        matrix = network._component_matrix(component)
        size = len(component.output_names)
        channels = []
        for index, name in enumerate(component.output_names):
            key = f"{component.label}.{name}"
            local = component._local_ports[index]
            coupling = zero if local is None else resolved_ports[local.label].coupling
            collapse = CollapseTerm(operator=coupling, rate=1.0, source=key, channel="port", frame_frequency=None)
            channels.append(SLHChannel(key=key, accessibility="exposed", collapse=collapse, coupling_operator=coupling))
        systems.append(
            ResolvedSLH(
                scattering=np.asarray(matrix, dtype=complex),
                hamiltonian=HamiltonianProgram(),
                channels=tuple(channels),
                support=np.ones((size, size), dtype=bool),
            )
        )
    return concatenate(*systems), {}


def _close_connections(network: PortNetwork, system: Any) -> Any:
    """Close every connection with feedback_reduce, following the merged channel keys it creates."""
    from quchip.engine import feedback_reduce

    current_input = {channel.key: channel.key for channel in system.channels}
    current_output = dict(current_input)
    for input_key, output_key in network._connections.items():
        source = current_output[".".join(output_key)]
        sink = current_input[".".join(input_key)]
        system = feedback_reduce(system, output=source, input=sink)
        merged = f"{source}->{sink}"
        for terminal, key in current_input.items():
            if key == source:
                current_input[terminal] = merged
        for terminal, key in current_output.items():
            if key == sink:
                current_output[terminal] = merged
    return system


def test_feedback_loop_matches_gough_james_reduction() -> None:
    """A lossy ring through a beam splitter compiles to the algebraic feedback reduction."""

    first = Resonator(freq=5.0, levels=3, label="a")
    second = Resonator(freq=5.0, levels=3, label="b")
    network = PortNetwork(label="ring")
    port_a = network.port("pa", target=first, rate=0.04)
    port_b = network.port("pb", target=second, rate=0.09)
    splitter = network.beam_splitter("bs", eta=0.36)
    phase = network.phase_shift("phi", phase=0.7)
    network.cascade(port_a, splitter.input_terminal("left"))
    network.cascade(splitter.output_terminal("left"), port_b, phase, splitter.input_terminal("right"))
    network.expose("ring", input=port_a.input, output=splitter.output_terminal("right"))
    compiled = Chip([first, second], port_network=network).resolve().slh

    bare = PortNetwork(label="bare")
    bare.port("pa", target=Resonator(freq=5.0, levels=3, label="a"), rate=0.04)
    bare.port("pb", target=Resonator(freq=5.0, levels=3, label="b"), rate=0.09)
    open_chip = Chip(list(device for port in bare.ports for device in port._targets), port_network=bare)
    oracle = _close_connections(network, _open_system(network, open_chip)[0])

    assert [channel.key for channel in compiled.channels] == ["ring"]
    np.testing.assert_allclose(compiled.S, oracle.S, atol=1e-12)
    np.testing.assert_allclose(compiled.L[0].to_dense(), oracle.L[0].to_dense(), atol=1e-12)
    compiled_h = sum(term.operator.to_dense() for term in compiled.H.static_terms if term.origin == "network")
    oracle_h = sum(term.operator.to_dense() for term in oracle.H.static_terms if term.origin == "network")
    np.testing.assert_allclose(compiled_h, oracle_h, atol=1e-12)
    assert compiled.feeds(0, 0)


def test_feedback_rejects_reference_cycle() -> None:
    """A delay inside an instantaneous loop is not Markovian and is refused by name."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="loop")
    port = network.port("p", target=resonator, rate=0.04)
    splitter = network.beam_splitter("bs", eta=0.5)
    cable = network.delay("cable", duration=1.0)
    network.cascade(port, splitter.input_terminal("left"))
    network.cascade(splitter.output_terminal("left"), cable.input_terminal("1"))
    network.cascade(cable.output_terminal("2"), splitter.input_terminal("right"))
    network.expose("out", input=port.input, output=splitter.output_terminal("right"))

    with pytest.raises(ValueError, match="cable.*feedback loop"):
        Chip([resonator], port_network=network).resolve()


def test_feedback_rejects_singular_concrete_loop() -> None:
    """A loop whose round trip is exactly unity has no instantaneous solution."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="loop")
    port = network.port("p", target=resonator, rate=0.04)
    line = network.through("line")
    network.connect(port.output, line.input)
    network.connect(line.output, port.input)

    with pytest.raises(ValueError, match="singular"):
        Chip([resonator], port_network=network).resolve()


def test_feedback_gain_is_jittable_and_differentiable() -> None:
    """A traced loop phase flows through the compiled ring gain."""
    jax = pytest.importorskip("jax")
    resonator = Resonator(freq=5.0, levels=2, label="r")

    def reflection(value: Any) -> Any:
        network = PortNetwork(label="ring")
        port = network.port("p", target=resonator, rate=0.04)
        splitter = network.beam_splitter("bs", eta=0.5)
        phase = network.phase_shift("phi", phase=value)
        network.cascade(port, splitter.input_terminal("left"))
        network.cascade(splitter.output_terminal("left"), phase, splitter.input_terminal("right"))
        network.expose("out", input=port.input, output=splitter.output_terminal("right"))
        slh = Chip([resonator], port_network=network).resolve().slh
        return jax.numpy.abs(slh.S[0, 0]) ** 2

    value, gradient = jax.jit(jax.value_and_grad(reflection))(jax.numpy.asarray(0.3))
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    np.testing.assert_allclose(float(value), float(reflection(0.3)), rtol=1e-6)


def _ring(eta: Any) -> Chip:
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="ring")
    port = network.port("p", target=resonator, rate=0.04)
    splitter = network.beam_splitter("bs", eta=eta)
    phase = network.phase_shift("phi", phase=0.3)
    network.cascade(port, splitter.input_terminal("left"))
    network.cascade(splitter.output_terminal("left"), phase, splitter.input_terminal("right"))
    network.expose("out", input=port.input, output=splitter.output_terminal("right"))
    return Chip([resonator], port_network=network)


def test_loop_support_is_structural_not_numerical() -> None:
    """Reachability through a loop follows the wiring, not the current coefficient values."""
    transparent = _ring(1.0).resolve().slh
    mixing = _ring(0.36).resolve().slh
    np.testing.assert_array_equal(transparent.support, mixing.support)
    assert transparent.feeds(0, 0)


def test_many_channel_loop_with_small_determinant_is_not_singular() -> None:
    """Five near-resonant loops in one component have a tiny determinant but a benign condition number."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="mixer")
    network.port("p", target=resonator, rate=0.04)
    detuning = 0.002
    names = tuple(str(index) for index in range(6))
    mixer = network.component("mix", scattering=np.diag(np.full(6, np.exp(1j * detuning))), terminals=names)
    for name in names[1:]:
        network.connect(mixer.output_terminal(name), mixer.input_terminal(name))
    network.expose("probe", input=mixer.input_terminal("0"), output=mixer.output_terminal("0"))

    resolved = Chip([resonator], port_network=network).resolve().slh

    assert abs(np.linalg.det(np.eye(5) - np.exp(1j * detuning) * np.eye(5))) < 1e-12
    probe = [channel.key for channel in resolved.channels].index("probe")
    np.testing.assert_allclose(resolved.S[probe, probe], np.exp(1j * detuning))


def _two_line_chip() -> Chip:
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=5.4, levels=2, label="b")
    network = PortNetwork(label="lines")
    port_a = network.port("pa", target=first, rate=0.02)
    port_b = network.port("pb", target=second, rate=0.03)
    loss = network.attenuator("loss", eta=0.5)
    cable = network.delay("cable", duration=1.0)
    network.link(port_a, loss)
    network.expose("line_a", at=loss.side(2))
    network.link(port_b, cable)
    network.expose("line_b", at=cable.side(2))
    return Chip([first, second], port_network=network)


def test_restrict_keeps_the_subgraphs_touching_selected_ports() -> None:
    """restrict() copies every component, connection, and plane reachable from the chosen ports."""
    chip = _two_line_chip()
    network = chip.port_network
    assert network is not None
    full = chip.resolve().slh

    line_a = network.restrict(["pa"])

    assert [port.label for port in line_a.ports] == ["pa"]
    assert {component.label for component in line_a.components} == {"pa", "loss"}
    assert [exposure.label for exposure in line_a.exposures] == ["line_a"]
    sub = Chip([Resonator(freq=5.0, levels=2, label="a")], port_network=line_a).resolve().slh
    keys = [channel.key for channel in full.channels]
    rows = [keys.index(channel.key) for channel in sub.channels]
    np.testing.assert_allclose(sub.S, np.asarray(full.S)[np.ix_(rows, rows)])
    assert "component.loss.eta" in line_a.parameters and "component.cable.duration" not in line_a.parameters

    line_b = network.restrict(["pb"])
    assert [exposure.label for exposure in line_b.exposures] == ["line_b"]
    resolved_b = Chip([Resonator(freq=5.4, levels=2, label="b")], port_network=line_b).resolve().slh
    assert [element.label for element in resolved_b.external_channels[0].reference.outbound] == ["cable"]


def test_restrict_rejects_subgraphs_spanning_other_ports() -> None:
    """A subgraph that also touches an unselected port cannot be cut without changing the dynamics."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=5.4, levels=2, label="b")
    network = PortNetwork(label="cascade")
    port_a = network.port("pa", target=first, rate=0.02)
    port_b = network.port("pb", target=second, rate=0.03)
    network.cascade(port_a, port_b)
    network.expose("feedline", input=port_a.input, output=port_b.output)

    with pytest.raises(ValueError, match="pb"):
        network.restrict(["pa"])
    with pytest.raises(KeyError):
        network.restrict(["missing"])
