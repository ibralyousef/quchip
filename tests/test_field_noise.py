"""Physical thermal fields use the input coupling, including directional routing."""
import json

import numpy as np
import pytest

from quchip import Chip, PortNetwork, Resonator


def test_warm_isolator_load_thermalizes_only_the_protected_device() -> None:
    """An isolator's warm termination feeds side 1 through the declared channel."""
    r = Resonator(freq=6.0, levels=12, label="r")
    net = PortNetwork(label="line")
    port = net.port("port", target=r, rate=0.03)
    iso = net.isolator("iso", occupation=0.3)
    net.link(port, iso)
    net.expose("readout", at=iso.port(2))
    chip = Chip([r], port_network=net)
    steady = chip.steadystate()
    diagonal = np.real(np.diag(steady.state.full()))
    expected = (0.3 / 1.3) ** np.arange(12)
    expected /= expected.sum()
    np.testing.assert_allclose(diagonal, expected, atol=1e-10)
    restored = Chip.from_dict(json.loads(json.dumps(chip.to_dict())))
    np.testing.assert_allclose(restored.steadystate().state.full(), steady.state.full(), atol=1e-10)
    cold = chip.with_params({"network.component.iso.occupation": 0.0})
    assert np.real(cold.steadystate().state.full()[0, 0]) == pytest.approx(1.0)


def test_matched_thermal_loss_and_temperature_agree() -> None:
    """A lossy line mixes a vacuum input with its thermal load occupation."""
    r = Resonator(freq=6.0, levels=12, label="r")
    net = PortNetwork(label="line")
    port = net.port("port", target=r, rate=0.03)
    att = net.attenuator("att", eta=0.25, temperature=300.0, noise_frequency=6.0)
    net.link(port, att)
    net.expose("readout", at=att.port(2))
    chip = Chip([r], port_network=net)
    state = chip.steadystate().state.full()
    # SI constants provide an independent occupation in the physical carrier frame.
    n = 0.75 / np.expm1(6.62607015e-34 * 6e9 / (1.380649e-23 * 0.300))
    expected = (n / (n + 1)) ** np.arange(12)
    expected /= expected.sum()
    np.testing.assert_allclose(np.diag(state), expected, atol=1e-10)


def test_thermal_state_declarations_reject_ambiguous_or_invalid_values() -> None:
    """Temperature needs a physical frequency and cannot accompany occupation."""
    for kwargs in ({"temperature": 20}, {"temperature": 20, "noise_frequency": 6, "occupation": 1},
                   {"occupation": -1}, {"temperature": -2, "noise_frequency": 6}):
        with pytest.raises(ValueError):
            PortNetwork().termination("load", **kwargs)


def test_thermal_equilibrium_output_is_flat_without_double_counting() -> None:
    """A warm source incident on a lossless cavity emerges with the same thermal spectrum."""
    from quchip import VNA
    r = Resonator(freq=6.0, levels=16, label="r")
    net = PortNetwork()
    port = net.port("p", target=r, rate=0.04)
    circ = net.circulator("circ")
    load = net.termination("warm", occupation=0.3)
    net.link(port, circ.port(2))
    net.link(load.port(1), circ.port(1))
    out = net.expose("out", at=circ.port(3))
    chip = Chip([r], port_network=net)
    result = VNA(chip).output_spectrum(out, frequencies=[-0.01, 0, 0.01])
    np.testing.assert_allclose(result.total_fluctuation_spectrum, 0.3, atol=2e-8)


def test_nonideal_unitary_component_uses_declared_thermal_loss_ports() -> None:
    """A unitary mixing component and a load provide a reusable finite-loss realization."""
    from quchip import VNA
    r = Resonator(freq=6.0, levels=3, label="r")
    net = PortNetwork()
    port = net.port("p", target=r, rate=0.04)
    # A reciprocal lossy two-port realized by a four-port unitary dilation.
    eta = 0.7
    t, loss = np.sqrt(eta), np.sqrt(1-eta)
    device = net.component("nonideal", scattering=[[0,t,loss,0], [t,0,0,loss],
                                                    [-loss,0,0,t], [0,-loss,t,0]],
                           terminals=("in", "out", "loss1", "loss2"))
    cold = net.termination("cold", occupation=0.0)
    warm = net.termination("warm", occupation=0.2)
    for output, input_, name in ((port.output, port.input, "in"),
                                 (cold.output_terminal("1"), cold.input_terminal("1"), "loss1"),
                                 (warm.output_terminal("1"), warm.input_terminal("1"), "loss2")):
        net.connect(output, device.input_terminal(name))
        net.connect(device.output_terminal(name), input_)
    out = net.expose("readout", input=device.input_terminal("out"), output=device.output_terminal("out"))
    result = VNA(Chip([r], port_network=net)).output_spectrum(out, frequencies=[0])
    np.testing.assert_allclose(result.total_fluctuation_spectrum, (1-eta)*0.2, atol=1e-10)


@pytest.mark.optional_backend
def test_thermal_collapse_coupling_rate_gradient() -> None:
    """Thermal channel lowering retains the analytic derivative with respect to coupling rate."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp
    mode = Resonator(freq=6.0, levels=8, label="r")
    net = PortNetwork()
    port = net.port("p", target=mode, rate=0.03)
    load = net.isolator("iso", occupation=0.3)
    net.link(port, load)
    net.expose("out", at=load.port(2))
    chip = Chip([mode], port_network=net, backend="dynamiqs")
    def weight(rate):
        term = chip.with_params({"port.p.rate": rate}).resolve().collapse_terms[-1]
        return term.rate * jnp.sum(jnp.abs(term.operator.to_dense())**2)
    value, gradient = jax.jit(jax.value_and_grad(weight))(0.03)
    np.testing.assert_allclose([value, gradient], [0.3*0.03*28, 0.3*28], atol=1e-9)


def test_declared_reverse_source_is_blocked_by_an_ideal_isolator() -> None:
    """Noise emitted into side 2 is absorbed by the isolator load, leaving the resonator cold."""
    mode = Resonator(freq=6.0, levels=8, label="r")
    net = PortNetwork()
    port = net.port("p", target=mode, rate=0.03)
    net.port("probe", target=mode, rate=0.01)
    iso = net.isolator("iso")
    warm = net.termination("reverse", occupation=1.0)
    net.link(port, iso, warm.port(1))
    chip = Chip([mode], port_network=net)
    np.testing.assert_allclose(np.diag(chip.steadystate().state.full()), [1,0,0,0,0,0,0,0], atol=1e-10)
