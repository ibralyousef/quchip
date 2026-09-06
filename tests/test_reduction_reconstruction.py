"""Reduction reconstruction preserves independent interaction owners."""
import numpy as np
import pytest

from quchip import Capacitive, Chip, CollapseChannel, DuffingTransmon, Resonator, eliminate


class NoisyCapacitive(Capacitive):
    def dissipation(self, a, b, p):
        return (CollapseChannel(a.a * b.I + a.I * b.a, .01, "collective"),)


def test_device_reduction_keeps_an_existing_direct_couplings_physics():
    a = DuffingTransmon(freq=5., anharmonicity=-.3, levels=3, label="a")
    b = DuffingTransmon(freq=5.3, anharmonicity=-.25, levels=3, label="b")
    bus = Resonator(freq=7., levels=3, label="bus")
    direct = NoisyCapacitive(a, b, g=.004, label="direct")
    source = Chip([a, b, bus], [direct, Capacitive(a, bus, g=.05), Capacitive(b, bus, g=.04)])
    result = eliminate(source, "bus")
    retained = result.chip.coupling("direct")
    assert type(retained) is NoisyCapacitive
    assert retained.g == pytest.approx(.004)
    assert len(result.chip.backend._collapse_operators(result.chip.resolve(frame="lab"))) == 1
    assert result.chip.coupling(result.effective_params["exchange"]["coupling"]).g == pytest.approx(
        float(result.effective_params["exchange"]["j_eff"]))


@pytest.mark.parametrize("direct_label", ["direct", "elim_fc"])
def test_direct_and_converted_pumps_keep_independent_targets_and_gains(direct_label):
    from quchip import ControlEquipment, FluxDrive, FluxTunableTransmon, ParametricDrive, QuantumSequence, Square
    from quchip import TunableCapacitive

    a = DuffingTransmon(freq=5., anharmonicity=-.3, levels=3, label="a")
    b = DuffingTransmon(freq=5.3, anharmonicity=-.25, levels=3, label="b")
    fc = FluxTunableTransmon(freq=7., anharmonicity=-.2, levels=3, label="fc")
    direct = TunableCapacitive(a, b, g_0=.004, label=direct_label)
    equipment = ControlEquipment([ParametricDrive(direct, label="pump"), FluxDrive(fc, label="flux")])
    source = Chip([a, b, fc], [direct, Capacitive(a, fc, g=.05), Capacitive(b, fc, g=.04)],
                  control_equipment=equipment)
    reduction = eliminate(source, "fc")
    reduced = reduction.chip
    lines = {line.label: line for line in reduced.control_equipment.lines}
    assert lines["pump"].target_label == direct_label
    mediated_label = reduction.effective_params["exchange"]["coupling"]
    assert lines["flux"].target_label == mediated_label != direct_label
    assert reduced.coupling(direct_label).g_0 == .004
    def response(chip, label):
        sequence = QuantumSequence(chip)
        baseline = np.asarray(sequence.hamiltonian().matrix(t=1.))
        sequence.schedule(label, envelope=Square(duration=2., amplitude=.01))
        return np.asarray(sequence.hamiltonian().matrix(t=1.)) - baseline

    expected_direct = reduction.mapping.project_operator(response(source, "pump"))
    np.testing.assert_allclose(response(reduced, "pump"), reduced.backend.to_array(expected_direct), atol=1e-12)
    expected_gain = float(reduction.effective_params["exchange"]["dJ_domega_c"])
    assert response(reduced, "flux")[3, 1] == pytest.approx(.01 * expected_gain, abs=1e-12)
