"""Continuous-wave input-output response through declared ports."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from quchip import (
    Capacitive,
    Chip,
    CrossKerr,
    DuffingTransmon,
    KerrCavity,
    Port,
    PortNetwork,
    Resonator,
    Sweep,
    VNA,
)


def _network(*ports: Port) -> PortNetwork:
    """Return the explicit identity boundary for the supplied ports."""
    return PortNetwork.from_ports(ports)


def _linear_resonator(*, kappa_in: float, kappa_out: float = 0.0):
    resonator = Resonator(freq=6.0, levels=8, label="r")
    input_port = Port(resonator, rate=kappa_in, label="in")
    ports = [input_port]
    output_port = None
    if kappa_out:
        output_port = Port(resonator, rate=kappa_out, label="out")
        ports.append(output_port)
    return resonator, input_port, output_port, Chip([resonator], port_network=_network(*ports))


def test_coherent_port_input_uses_standard_slh_hamiltonian_sign() -> None:
    """A coherent source composes as i(beta* L - beta L-dagger)."""
    from quchip.engine.input_output import (
        add_port_inputs,
        port_operators,
        resolve_stationary_engine,
    )

    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    engine = resolve_stationary_engine(chip, ((input_port.label, 6.0),))
    beta = 0.02 + 0.01j
    coupling = port_operators(engine, chip.backend)[input_port.label].to_dense()

    driven = add_port_inputs(engine, chip.backend, ((input_port.label, 6.0, beta),))

    expected = 1j * (np.conj(beta) * coupling - beta * coupling.conj().T)
    np.testing.assert_allclose(driven.static_terms[-1].operator.to_dense(), expected)


def test_coherent_port_input_leaves_resolved_slh_input_free() -> None:
    """Binding beta adds solve physics beside the immutable resolved SLH value."""
    from quchip.engine.input_output import add_port_inputs, resolve_stationary_engine

    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    engine = resolve_stationary_engine(chip, ((input_port.label, 6.0),))

    driven = add_port_inputs(
        engine,
        chip.backend,
        ((input_port.label, 6.0, 0.02 + 0.01j),),
    )

    assert driven.slh is engine.slh
    assert driven.slh.H == engine.slh.H
    assert len(driven.static_terms) == len(engine.static_terms) + 1


def test_output_field_uses_standard_slh_plus_sign() -> None:
    """The reported field is b_out = beta I + L at the reference plane."""
    from quchip.analysis.vna import _output_field_matrix
    from quchip.engine.input_output import port_operators, resolve_stationary_engine

    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    engine = resolve_stationary_engine(chip, ((input_port.label, 6.0),))
    coupling = port_operators(engine, chip.backend)[input_port.label]
    beta = 0.02 + 0.01j

    output = _output_field_matrix(coupling, beta, np)

    expected = beta * np.eye(coupling.shape[0], dtype=complex) + coupling.to_dense()
    np.testing.assert_allclose(output, expected)


def test_one_sided_small_signal_reflection_matches_analytic_response() -> None:
    resonator, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    frequencies = np.array([5.98, 6.0, 6.03])

    result = VNA(chip, planes=[input_port]).sweep(frequencies)

    detuning = 2 * np.pi * (resonator.freq - frequencies)
    expected = 1.0 - 0.04 / (0.02 + 1j * detuning)
    np.testing.assert_allclose(result.s11, expected, atol=2e-8)
    np.testing.assert_allclose(result.s("in", "in"), expected, atol=2e-8)
    np.testing.assert_allclose(result.frequencies, frequencies)


def test_two_sided_transmission_is_unit_magnitude_on_resonance() -> None:
    _, input_port, output_port, chip = _linear_resonator(kappa_in=0.03, kappa_out=0.03)
    assert output_port is not None

    result = VNA(chip, planes=[input_port, output_port]).sweep([6.0])

    np.testing.assert_allclose(result.s11, [0.0], atol=2e-8)
    np.testing.assert_allclose(result.s21, [-1.0], atol=2e-8)
    np.testing.assert_allclose(np.abs(result.s11) ** 2 + np.abs(result.s21) ** 2, [1.0], atol=2e-8)


def test_vna_returns_the_full_plane_matrix_in_one_sweep() -> None:
    """All S_ji between the selected planes come from one solve per frequency."""
    resonator, input_port, output_port, chip = _linear_resonator(kappa_in=0.04, kappa_out=0.02)
    frequencies = np.array([5.98, 6.0, 6.03])

    result = VNA(chip).sweep(frequencies)
    single = VNA(chip, planes=[output_port]).sweep(frequencies)

    assert result.planes == ("in", "out")
    assert result.matrix.shape == (3, 2, 2)
    np.testing.assert_allclose(result.s21, result.matrix[:, 1, 0])
    np.testing.assert_allclose(result.s("in", output_port), result.matrix[:, 0, 1])
    np.testing.assert_allclose(result.s(output_port, output_port), single.s11, atol=1e-12)
    power = np.sum(np.abs(result.matrix) ** 2, axis=1)
    np.testing.assert_allclose(power, np.ones((3, 2)), atol=1e-8)
    np.testing.assert_allclose(np.asarray(result), result.matrix)
    with pytest.raises(AttributeError, match="s21"):
        _ = single.s21
    with pytest.raises(TypeError, match="sequence"):
        VNA(chip, planes="in")


def test_stationary_matrix_matches_mode_space_matrix() -> None:
    """The multi-source Liouvillian path agrees with the passive-linear matrix."""
    _, _, _, chip = _linear_resonator(kappa_in=0.04, kappa_out=0.02)
    frequencies = np.array([5.99, 6.0, 6.01])
    linear = VNA(chip).sweep(frequencies)
    general = VNA(chip).sweep(frequencies, options={"method": "direct"})

    assert {item["solver"] for item in linear.diagnostics} == {"linear_response"}
    assert {item["solver"] for item in general.diagnostics} == {"stationary_resolvent"}
    np.testing.assert_allclose(general.matrix, linear.matrix, atol=2e-8)


def test_hidden_dilation_channels_carry_probe_and_pump_fields() -> None:
    """Coherent sources sum conj(S) L over every channel, hidden vacuum outputs included."""
    resonator = Resonator(freq=6.0, levels=8, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.link(port, loss)
    network.expose("readout", at=loss.side(2))
    chip = Chip([resonator], port_network=network)
    frequencies = np.array([5.99, 6.0, 6.01])

    linear = VNA(chip).sweep(frequencies)
    general = VNA(chip).sweep(frequencies, options={"method": "direct"})
    np.testing.assert_allclose(general.matrix, linear.matrix, atol=2e-8)

    bare = Resonator(freq=6.0, levels=8, label="r")
    bare_network = PortNetwork(label="bare")
    bare_network.port("coupler", target=bare, rate=0.04)
    bare_chip = Chip([bare], port_network=bare_network)
    attenuated = VNA(chip, planes=["readout"])
    attenuated.pump("readout", freq=6.0, amplitude=0.05)
    direct = VNA(bare_chip, planes=["coupler"])
    direct.pump("coupler", freq=6.0, amplitude=0.8 * 0.05)
    lossy_state = attenuated._stationary_output("readout", None)[1].state.full()
    direct_state = direct._stationary_output("coupler", None)[1].state.full()
    np.testing.assert_allclose(lossy_state, direct_state, atol=1e-8)


def _lowpass(frequency, *, cutoff, order):
    return 1.0 / (1.0 + 1j * (frequency / cutoff) ** order)


def test_filter_section_is_exact_for_continuous_waves_and_sweepable() -> None:
    """A reciprocal filter multiplies both legs by H(f); its parameters are chip paths."""
    resonator = Resonator(freq=6.0, levels=8, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=0.04)
    lowpass = network.filter("lowpass", transfer=_lowpass, cutoff=6.5, order=2)
    network.link(port, lowpass)
    network.expose("readout", at=lowpass.side(2))
    chip = Chip([resonator], port_network=network)
    frequencies = np.array([5.98, 6.0, 6.02])

    bare = Resonator(freq=6.0, levels=8, label="r")
    bare_network = PortNetwork(label="bare")
    bare_port = bare_network.port("coupler", target=bare, rate=0.04)
    bare_chip = Chip([bare], port_network=bare_network)

    filtered = VNA(chip).sweep(frequencies)
    reference = VNA(bare_chip, planes=[bare_port]).sweep(frequencies)
    expected = _lowpass(frequencies, cutoff=6.5, order=2) ** 2 * reference.s11
    np.testing.assert_allclose(filtered.s11, expected, atol=1e-10)
    np.testing.assert_allclose(
        VNA(chip).sweep(frequencies, options={"method": "direct"}).s11, expected, atol=2e-8
    )

    rebound = VNA(chip).sweep(frequencies, Sweep([6.5, 8.0], name="network.component.lowpass.cutoff"))
    np.testing.assert_allclose(rebound.s11[0], expected, atol=1e-10)
    np.testing.assert_allclose(
        rebound.s11[1], _lowpass(frequencies, cutoff=8.0, order=2) ** 2 * reference.s11, atol=1e-10
    )
    with pytest.raises(TypeError, match="lowpass"):
        chip.to_dict()


def test_filter_section_must_be_passive() -> None:
    """A concretely evaluated |H| above one is gain and is rejected."""
    resonator = Resonator(freq=6.0, levels=4, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=0.04)
    amplifier = network.filter("gain", transfer=lambda frequency, *, gain: gain, gain=2.0)
    network.link(port, amplifier)
    network.expose("readout", at=amplifier.side(2))

    with pytest.raises(ValueError, match="passive"):
        VNA(Chip([resonator], port_network=network)).sweep([6.0])


def test_output_spectrum_is_filtered_at_the_offset_frequency() -> None:
    """An outbound filter scales the fluctuation spectrum by |H(f_frame + nu)|^2."""

    def build(with_filter: bool):
        resonator = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
        network = PortNetwork(label="line")
        port = network.port("coupler", target=resonator, rate=0.04)
        if with_filter:
            section = network.filter("lowpass", transfer=_lowpass, cutoff=6.02, order=1)
            network.link(port, section)
            network.expose("readout", at=section.side(2))
        else:
            network.expose("readout", at=port)
        vna = VNA(Chip([resonator], port_network=network))
        inbound = 1.0 if with_filter else abs(_lowpass(6.0, cutoff=6.02, order=1))
        vna.pump("readout", freq=6.0, amplitude=0.02 * inbound)
        return vna

    offsets = np.array([-0.02, 0.0, 0.03])
    filtered = build(True).output_spectrum("readout", frequencies=offsets)
    plain = build(False).output_spectrum("readout", frequencies=offsets)
    gain = np.abs(_lowpass(6.0 + offsets, cutoff=6.02, order=1)) ** 2
    np.testing.assert_allclose(filtered.fluctuation_spectrum, gain * plain.fluctuation_spectrum, rtol=1e-8)


def test_output_notch_filter_keeps_the_sideband_spectrum() -> None:
    """A filter that nulls the carrier still passes the sidebands with |H(f_c + nu)|^2."""

    def notch(frequency, *, center, width):
        detuning = (frequency - center) / width
        return 1j * detuning / (1.0 + 1j * detuning)

    def build(with_filter: bool):
        qubit = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
        network = PortNetwork(label="fridge")
        port = network.port("coupler", target=qubit, rate=0.04)
        circulator = network.circulator("circ")
        network.link(port, circulator.side(2))
        network.expose("drive", at=circulator.side(1))
        if with_filter:
            section = network.filter("notch", transfer=notch, center=6.0, width=0.01)
            network.link(circulator.side(3), section)
            network.expose("readout", at=section.side(2))
        else:
            network.expose("readout", at=circulator.side(3))
        vna = VNA(Chip([qubit], port_network=network))
        vna.pump("drive", freq=6.0, amplitude=0.02)
        return vna

    offsets = np.array([-0.02, 0.0, 0.03])
    filtered = build(True).output_spectrum("readout", frequencies=offsets)
    plain = build(False).output_spectrum("readout", frequencies=offsets)
    gain = np.abs(notch(6.0 + offsets, center=6.0, width=0.01)) ** 2
    np.testing.assert_allclose(filtered.fluctuation_spectrum, gain * plain.fluctuation_spectrum, atol=1e-12)
    assert filtered.fluctuation_spectrum[1] == pytest.approx(0.0, abs=1e-12)
    assert filtered.fluctuation_spectrum[2] > 0.0


def test_coupling_free_output_takes_its_carrier_from_the_feeding_tone() -> None:
    """A circulator's drive-side output has no coupling; its filter follows the unique feeding tone."""

    def tilt(frequency, *, slope):
        return 0.5 * np.exp(1j * slope * frequency)

    def build(pumped: bool, *, delay_only: bool = False):
        qubit = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
        network = PortNetwork(label="fridge")
        port = network.port("coupler", target=qubit, rate=0.04)
        circulator = network.circulator("circ")
        line = (
            network.delay("line", duration=0.3)
            if delay_only
            else network.filter("line", transfer=tilt, slope=0.7)
        )
        network.link(line, circulator.side(1))
        network.link(port, circulator.side(2))
        network.expose("drive", at=line.side(1))
        network.expose("readout", at=circulator.side(3))
        vna = VNA(Chip([qubit], port_network=network))
        if pumped:
            vna.pump("readout", freq=6.0, amplitude=0.03)
        return vna

    spectrum = build(True).output_spectrum("drive", frequencies=np.array([0.0]))
    assert spectrum.coherent_flux == pytest.approx(abs(tilt(6.0, slope=0.7)) ** 2 * 0.03**2)
    with pytest.raises(ValueError, match="carrier"):
        build(False).output_spectrum("drive", frequencies=np.array([0.0]))
    delayed = build(True, delay_only=True).output_spectrum("drive", frequencies=np.array([0.0]))
    assert delayed.coherent_flux == pytest.approx(0.03**2)
    with pytest.raises(ValueError, match="carrier"):
        build(False, delay_only=True).output_spectrum("drive", frequencies=np.array([0.0]))
    assert build(False).output_spectrum("readout", frequencies=np.array([0.0])).coherent_flux == pytest.approx(0.0)


def test_distinct_tones_stay_valid_with_traced_network_scattering() -> None:
    """Reachability comes from the compiled structure, not from traced S values."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    import jax.numpy as jnp

    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    resonator = Resonator(freq=6.0, levels=4, label="r")
    network = PortNetwork(label="fridge")
    qubit_port = network.port("qubit_port", target=qubit, rate=0.04)
    readout_port = network.port("readout_port", target=resonator, rate=0.03)
    loss = network.attenuator("loss", eta=0.5)
    network.link(readout_port, loss)
    network.expose("readout", at=loss.side(2))
    chip = Chip(
        [qubit, resonator],
        [CrossKerr(qubit, resonator, chi=-0.03)],
        port_network=network,
        backend="dynamiqs",
    )

    def response(eta):
        vna = VNA(chip.with_params({"network.component.loss.eta": eta}), planes=["readout"])
        vna.pump(qubit_port, freq=5.0, amplitude=0.02)
        return jnp.abs(vna.sweep([6.0]).s11[0])

    assert jnp.isfinite(jax.jit(response)(jnp.asarray(0.5)))


def test_vna_uses_network_exposure_labels_and_scattering_background() -> None:
    """VNA queries named network exposures and retains direct scattering."""
    resonator = Resonator(freq=6.0, levels=6, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    phase = network.phase_shift("phase", phase=np.pi / 2)
    network.cascade(port, phase)
    network.expose("readout", input=port.input, output=phase.output)
    chip = Chip([resonator], port_network=network)

    result = VNA(chip, planes=["readout"]).sweep([5.98, 6.0, 6.02])

    detuning = 2 * np.pi * (resonator.freq - np.asarray([5.98, 6.0, 6.02]))
    expected = 1j * (1.0 - 0.04 / (0.02 + 1j * detuning))
    np.testing.assert_allclose(result.s11, expected, atol=2e-8)


def test_vna_reference_delay_is_reciprocal() -> None:
    """VNA phase accumulates the reciprocal external reference-plane delay."""
    def response(delay: float) -> complex:
        resonator = Resonator(freq=6.0, levels=5, label="r")
        network = PortNetwork(label="line")
        port = network.port("chip_port", target=resonator, rate=0.04)
        cable = network.delay("cable", duration=delay)
        network.link(port, cable)
        network.expose("readout", at=cable.side(2))
        chip = Chip([resonator], port_network=network)
        return complex(VNA(chip, planes=["readout"]).sweep([6.01]).s11[0])

    delay = 0.125
    expected_phase = np.exp(1j * 2.0 * 2.0 * np.pi * 6.01 * delay)
    assert response(delay) == pytest.approx(expected_phase * response(0.0), abs=2e-8)


@pytest.mark.parametrize("internal_rate,external_rate", [(0.04, 0.02), (0.02, 0.02), (0.01, 0.03)])
def test_resonance_distinguishes_undercritical_and_overcoupling(
    internal_rate: float,
    external_rate: float,
) -> None:
    resonator = Resonator(freq=6.0, levels=5, label="r", T1=1.0 / internal_rate)
    port = Port(resonator, rate=external_rate, label="p")
    result = VNA(
        Chip([resonator], port_network=_network(port)),
        planes=[port],
    ).sweep([6.0])

    expected = (internal_rate - external_rate) / (internal_rate + external_rate)
    np.testing.assert_allclose(result.s11, [expected], atol=2e-8)


def test_vna_probe_has_no_finite_amplitude_mode() -> None:
    """Finite-power spectroscopy belongs to external-plane input simulations."""
    assert "amplitude" not in inspect.signature(VNA.sweep).parameters


def test_chip_parameter_sweeps_are_vna_axes() -> None:
    """Ordinary chip parameter sweeps rebind the chip per grid point beside tone axes."""
    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    frequencies = np.array([5.98, 6.0, 6.02])
    vna = VNA(chip, planes=[input_port])

    result = vna.sweep(frequencies, Sweep([5.98, 6.02], name="r.freq"))

    assert result.shape == (2, 3)
    assert result.axis_names == ("r.freq", "frequency")
    for row, freq in zip(result.s11, [5.98, 6.02], strict=True):
        shifted = VNA(chip.with_params({"r.freq": freq}), planes=[input_port]).sweep(frequencies)
        np.testing.assert_allclose(row, shifted.s11, atol=1e-12)


def test_chip_paths_resembling_tone_keys_still_rebind() -> None:
    """Chip bindings are classified by exact tone keys, not by a label prefix."""
    resonator = Resonator(freq=6.0, levels=6, label="__vna_tone_0")
    port = Port(resonator, rate=0.04, label="p")
    chip = Chip([resonator], port_network=_network(port))

    result = VNA(chip).sweep(np.array([5.9, 6.1]), Sweep([5.9, 6.1], name="__vna_tone_0.freq"))

    np.testing.assert_allclose(np.abs(result.s11[0, 0]), np.abs(result.s11[1, 1]), atol=1e-8)
    assert not np.allclose(result.s11[0, 0], result.s11[0, 1])


def test_vna_rejects_unknown_paths_and_foreign_tone_axes() -> None:
    """Unknown parameter paths and another VNA's tone axes fail loudly."""
    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    vna = VNA(chip, planes=[input_port])

    with pytest.raises(KeyError, match="r.frequency"):
        vna.sweep([6.0], Sweep([5.9], name="r.frequency"))
    other = VNA(chip, planes=[input_port]).pump(input_port, freq=5.0, amplitude=0.01)
    with pytest.raises(ValueError, match="this VNA"):
        vna.sweep([6.0], other.vary("freq", [4.9]))


def test_chip_and_pump_axes_zip_together() -> None:
    """A zipped chip parameter and pump axis step through the grid element by element."""
    readout = Resonator(freq=6.0, levels=4, label="readout")
    auxiliary = Resonator(freq=5.0, levels=3, label="aux")
    readout_port = Port(readout, rate=0.03, label="readout_port")
    pump_port = Port(auxiliary, rate=0.04, label="pump_port")
    chip = Chip([readout, auxiliary], port_network=_network(readout_port, pump_port))
    vna = VNA(chip, planes=[readout_port])
    pump = vna.pump(pump_port, freq=5.0, amplitude=0.02)

    result = vna.sweep(
        np.array([5.99, 6.0]),
        vna.zip(pump.vary("amplitude", [0.01, 0.02]), Sweep([0.03, 0.05], name="port.readout_port.rate")),
    )

    assert result.shape == (2, 2)
    assert result.axis_names == ("pump_port.amplitude/port.readout_port.rate", "frequency")
    assert np.all(np.isfinite(result.s11))


def test_vna_rejects_duplicate_public_axis_names() -> None:
    """Custom axis names cannot collide with each other or with the frequency axis."""
    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    vna = VNA(chip, planes=[input_port])
    pump = vna.pump(input_port, freq=5.0, amplitude=0.01)

    with pytest.raises(ValueError, match="unique"):
        vna.sweep(
            [6.0, 6.01],
            vna.zip(pump.vary("freq", [4.9, 5.1], name="power"), pump.vary("amplitude", [0.01, 0.02], name="power")),
        )
    with pytest.raises(ValueError, match="unique"):
        vna.sweep([6.0, 6.01], pump.vary("amplitude", [0.01, 0.02], name="frequency"))
    with pytest.raises(ValueError, match="unique"):
        vna.sweep(
            6.0,
            vna.zip(pump.vary("freq", [4.9, 5.1], name="gain"), pump.vary("amplitude", [0.01, 0.02], name="phase")),
            pump.vary("amplitude", [0.03, 0.04], name="gain/phase"),
        )


def test_vna_result_does_not_depend_on_chip_default_frame() -> None:
    """VNA resolves the stationary tone frame explicitly."""
    def response(frame):
        resonator = Resonator(freq=6.0, levels=5, label="r")
        port = Port(resonator, rate=0.04, label="p")
        chip = Chip([resonator], port_network=_network(port), frame=frame)
        return VNA(chip, planes=[port]).sweep([5.98, 6.0, 6.02]).s11

    np.testing.assert_allclose(response("lab"), response("rotating"), atol=2e-8)


def test_one_carrier_probes_passive_modes_behind_the_port() -> None:
    """The probe frame covers a passive filter-readout network, not only the port target."""
    readout = Resonator(freq=6.0, levels=2, label="readout")
    purcell_filter = Resonator(freq=6.02, levels=2, label="filter")
    feedline = Port(purcell_filter, rate=0.02, label="feedline")
    chip = Chip(
        [purcell_filter, readout],
        couplings=[Capacitive(purcell_filter, readout, g=0.01)],
        port_network=_network(feedline),
    )

    result = VNA(chip, planes=[feedline]).sweep([5.99, 6.0, 6.02])

    assert result.s11.shape == (3,)
    assert np.all(np.isfinite(result.s11))


def test_fixed_tone_variation_uses_pump_axes_without_new_drive_physics() -> None:
    readout = Resonator(freq=6.0, levels=4, label="readout")
    auxiliary = Resonator(freq=5.0, levels=3, label="aux")
    readout_port = Port(readout, rate=0.03, label="readout_port")
    pump_port = Port(auxiliary, rate=0.04, label="pump_port")
    chip = Chip([readout, auxiliary], port_network=_network(readout_port, pump_port))
    vna = VNA(chip, planes=[readout_port])
    pump = vna.pump(pump_port, freq=5.0, amplitude=0.02)

    result = vna.sweep(
        np.array([5.99, 6.0]),
        pump.vary("freq", np.array([4.98, 5.02, 5.04]), name="pump_freq"),
    )

    assert result.shape == (3, 2)
    assert result.axis_names == ("pump_freq", "frequency")


def test_distinct_stationary_tones_on_one_mode_require_time_evolution() -> None:
    resonator = Resonator(freq=6.0, levels=4, label="r")
    probe = Port(resonator, rate=0.02, label="probe")
    pump_port = Port(resonator, rate=0.02, label="pump")
    chip = Chip([resonator], port_network=_network(probe, pump_port))
    vna = VNA(chip, planes=[probe])
    vna.pump(pump_port, freq=5.9, amplitude=0.01)

    with pytest.raises(ValueError, match="QuantumSequence"):
        vna.sweep([6.0])


def test_qutip_and_dynamiqs_vna_response_agree() -> None:
    pytest.importorskip("dynamiqs")

    def response(backend: str):
        resonator = Resonator(freq=6.0, levels=5, label="r")
        input_port = Port(resonator, rate=0.02, label="in")
        output_port = Port(resonator, rate=0.03, label="out")
        chip = Chip(
            [resonator],
            port_network=_network(input_port, output_port),
            backend=backend,
        )
        return VNA(chip, planes=[input_port, output_port]).sweep([5.98, 6.0, 6.02]).s21

    np.testing.assert_allclose(np.asarray(response("dynamiqs")), response("qutip"), atol=2e-8)


def test_passive_linear_vna_uses_mode_space_and_matches_one_port_reflection() -> None:
    """Passive harmonic scattering uses the exact mode-space transfer function."""
    resonator = Resonator(freq=6.0, levels=8, label="r")
    port = Port(resonator, rate=0.04, label="readout")
    chip = Chip([resonator], port_network=_network(port))
    frequencies = np.asarray([5.98, 6.0, 6.02])

    result = VNA(chip, planes=[port]).sweep(frequencies)

    detuning = 2 * np.pi * (resonator.freq - frequencies)
    expected = 1.0 - port.rate / (port.rate / 2.0 + 1j * detuning)
    np.testing.assert_allclose(result.s11, expected, atol=2e-12)
    assert {diagnostic["solver"] for diagnostic in result.diagnostics} == {
        "linear_response"
    }


def test_eight_resonator_cascade_matches_exact_series_product() -> None:
    """An eight-mode feedline stays in mode space and reproduces SLH series response."""
    resonances = np.asarray([6.42, 6.505, 6.595, 6.69, 6.79, 6.895, 7.005, 7.12])
    external_rates = np.asarray([0.0060, 0.0082, 0.0052, 0.0105, 0.0071, 0.0120, 0.0090, 0.0066])
    qualities = np.asarray([28_000, 42_000, 22_000, 55_000, 31_000, 47_000, 26_000, 60_000])
    phases = np.asarray([0.18, 0.27, 0.16, 0.31, 0.22, 0.28, 0.19])
    resonators = [
        Resonator(
            freq=frequency,
            levels=32,
            internal_quality_factor=quality,
            label=f"r{index}",
        )
        for index, (frequency, quality) in enumerate(zip(resonances, qualities, strict=True))
    ]
    network = PortNetwork(label="feedline")
    ports = [
        network.port(f"p{index}", target=resonator, rate=rate)
        for index, (resonator, rate) in enumerate(zip(resonators, external_rates, strict=True))
    ]
    previous = ports[0]
    for index, (phase, following) in enumerate(zip(phases, ports[1:], strict=True)):
        section = network.phase_shift(f"section{index}", phase=phase)
        network.cascade(previous, section, following)
        previous = following
    input_line = network.delay("input_line", duration=0.08)
    output_line = network.delay("output_line", duration=0.08)
    network.connect(input_line.output_terminal("2"), ports[0].input)
    network.connect(ports[-1].output, output_line.input_terminal("1"))
    network.expose(
        "readout",
        input=input_line.input_terminal("1"),
        output=output_line.output_terminal("2"),
    )
    frequencies = np.asarray([6.42, 6.69, 7.12])

    result = VNA(
        Chip(resonators, port_network=network),
        planes=["readout"],
    ).sweep(frequencies)

    expected = np.exp(1j * np.sum(phases)) * np.exp(1j * 4 * np.pi * frequencies * 0.08)
    for resonance, external_rate, quality in zip(resonances, external_rates, qualities, strict=True):
        internal_rate = 2 * np.pi * resonance / quality
        expected *= 1.0 - external_rate / (
            0.5 * (external_rate + internal_rate) + 1j * 2 * np.pi * (resonance - frequencies)
        )
    np.testing.assert_allclose(result.s11, expected, atol=2e-11)
    assert all(diagnostic["mode_count"] == 8 for diagnostic in result.diagnostics)


def test_coupled_mode_response_matches_general_liouvillian_fallback() -> None:
    """Mode-space scattering agrees with the general solver for a coupled linear chip."""
    frequencies = np.asarray([5.97, 6.0, 6.04])

    def response(*, explicit_operator: bool):
        first = Resonator(freq=6.0, levels=3, label="a")
        second = Resonator(freq=6.035, levels=3, label="b")
        operator = (
            np.diag(np.sqrt(np.arange(1, first.levels)), k=1).astype(complex)
            if explicit_operator
            else None
        )
        port = Port(first, rate=0.035, operator=operator, label="readout")
        chip = Chip(
            [first, second],
            [Capacitive(first, second, g=0.012)],
            port_network=_network(port),
        )
        return VNA(chip, planes=[port]).sweep(frequencies)

    linear = response(explicit_operator=False)
    general = response(explicit_operator=True)

    np.testing.assert_allclose(linear.s11, general.s11, atol=2e-10)
    assert {item["solver"] for item in linear.diagnostics} == {"linear_response"}
    assert {item["solver"] for item in general.diagnostics} == {"stationary_resolvent"}


def test_nonlinear_hamiltonian_retains_stationary_fallback() -> None:
    """A structurally nonlinear mode remains on the general stationary solver."""
    cavity = KerrCavity(freq=6.0, kerr=0.02, levels=3, label="c")
    port = Port(cavity, rate=0.04, label="readout")

    result = VNA(
        Chip([cavity], port_network=_network(port)),
        planes=[port],
    ).sweep([6.0])

    assert {item["solver"] for item in result.diagnostics} == {"stationary_resolvent"}


def test_dynamiqs_linear_response_is_jittable_and_differentiable() -> None:
    """The mode-space VNA path preserves JIT and gradients through chip parameters."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    resonator = Resonator(freq=6.0, levels=16, label="r")
    port = Port(resonator, rate=0.04, label="readout")
    chip = Chip([resonator], port_network=_network(port), backend="dynamiqs")

    def reflection(resonance):
        shifted = chip.with_params({"r.freq": resonance})
        return jnp.real(VNA(shifted, planes=["readout"]).sweep([6.01]).s11[0])

    value, gradient = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(6.0))

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_dynamiqs_pumped_small_signal_response_is_jittable_and_differentiable() -> None:
    """Pumped Dynamiqs small-signal response is JIT-safe and differentiable."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    resonator = Resonator(freq=6.0, levels=3, label="r")
    port = Port(resonator, rate=0.04, label="p")
    chip = Chip([resonator], port_network=_network(port), backend="dynamiqs")

    def reflection(amplitude):
        vna = VNA(chip, planes=[port])
        vna.pump(port, freq=6.0, amplitude=amplitude)
        return jnp.real(vna.sweep([6.0]).s11[0])

    value = jax.jit(reflection)(jnp.asarray(0.01))
    gradient = jax.grad(reflection)(jnp.asarray(0.01))

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_dynamiqs_explicit_port_frequency_is_jittable_and_differentiable(
) -> None:
    """Small-signal VNA preserves an explicit port's traced frequency."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    resonator = Resonator(freq=6.0, levels=3, label="r")
    lowering = np.diag(np.sqrt(np.arange(1, 3)), k=1).astype(complex)
    port = Port(resonator, rate=0.04, operator=lowering, label="p")
    vna = VNA(
        Chip([resonator], port_network=_network(port), backend="dynamiqs"),
        planes=[port],
    )

    def reflection(frequency):
        return jnp.real(vna.sweep(jnp.asarray([frequency])).s11[0])

    value, gradient = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(6.01))

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_resonantly_driven_two_level_population_matches_optical_bloch_solution() -> None:
    decay_rate = 0.05
    amplitude = 0.03
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    port = Port(qubit, rate=decay_rate, label="drive")
    chip = Chip([qubit], port_network=_network(port))

    vna = VNA(chip, planes=[port])
    vna.pump(port, freq=5.0, amplitude=amplitude)
    state = np.asarray(vna._stationary_output(port.label, None)[1].state.full())
    excited_population = np.real(state[1, 1])
    expected = 4 * amplitude**2 / (decay_rate + 8 * amplitude**2)

    np.testing.assert_allclose(excited_population, expected, atol=2e-8)


def test_two_tone_cross_kerr_model_produces_a_pump_frequency_axis() -> None:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    resonator = Resonator(freq=6.0, levels=4, label="r")
    qubit_port = Port(qubit, rate=0.04, label="qubit_port")
    readout_port = Port(resonator, rate=0.03, label="readout_port")
    chip = Chip(
        [qubit, resonator],
        [CrossKerr(qubit, resonator, chi=-0.03)],
        port_network=_network(qubit_port, readout_port),
    )
    vna = VNA(chip, planes=[readout_port])
    pump = vna.pump(qubit_port, freq=5.0, amplitude=0.04)

    result = vna.sweep(
        np.array([5.99, 6.0]),
        pump.vary("freq", np.array([4.97, 5.0, 5.03])),
    )

    assert result.s11.shape == (3, 2)
    assert result.axis_names == ("qubit_port.freq", "frequency")
    assert not np.allclose(result.s11[0], result.s11[1])


def test_two_tone_probe_frame_propagates_through_passive_filter_network() -> None:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    resonator = Resonator(freq=6.0, levels=2, label="r")
    purcell_filter = Resonator(freq=6.02, levels=2, label="filter")
    qubit_port = Port(qubit, rate=0.04, label="qubit_port")
    feedline = Port(purcell_filter, rate=0.20, label="feedline")
    chip = Chip(
        [qubit, resonator, purcell_filter],
        [
            CrossKerr(qubit, resonator, chi=-0.003),
            Capacitive(purcell_filter, resonator, g=0.010),
        ],
        port_network=_network(qubit_port, feedline),
    )
    vna = VNA(chip, planes=[feedline])
    pump = vna.pump(qubit_port, freq=5.0, amplitude=0.02)

    result = vna.sweep(
        np.array([5.99, 6.00]),
        pump.vary("freq", np.array([4.98, 5.00, 5.02])),
    )

    assert result.s11.shape == (3, 2)
    assert result.axis_names == ("qubit_port.freq", "frequency")
    assert np.all(np.isfinite(result.s11))


def test_nonlinear_stationary_state_matches_long_time_master_equation() -> None:
    import qutip

    amplitude = 0.05
    cavity = KerrCavity(freq=6.0, kerr=0.03, levels=8, label="c")
    port = Port(cavity, rate=0.05, label="p")
    chip = Chip([cavity], port_network=_network(port), backend="qutip")
    vna = VNA(chip, planes=[port])
    vna.pump(port, freq=6.0, amplitude=amplitude)
    engine = chip.resolve(frame={"c": 6.0})
    backend = chip.backend
    hamiltonian = sum(
        (
            term.coefficient * backend.from_canonical_operator(term.operator)
            for term in engine.static_terms
        ),
        start=0,
    )
    port_term = engine.port_terms[0]
    coupling = np.exp(1j * port_term.phase) * np.sqrt(port_term.rate) * backend.from_canonical_operator(
        port_term.operator
    )
    hamiltonian = hamiltonian + 1j * (amplitude.conjugate() * coupling - amplitude * coupling.dag())
    collapse = [
        np.sqrt(term.rate) * backend.from_canonical_operator(term.operator)
        for term in engine.collapse_terms
    ]
    initial = qutip.ket2dm(qutip.basis(cavity.levels, 0))
    evolved = qutip.mesolve(
        hamiltonian,
        initial,
        [0.0, 1000.0],
        collapse,
        options={"method": "diag"},
    ).states[-1]

    np.testing.assert_allclose(vna._stationary_output(port.label, None)[1].state.full(), evolved.full(), atol=2e-7)
