"""Automatic constant-generator solver selection at the backend boundary."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import ChargeDrive, Chip, DuffingTransmon, PortNetwork, QuantumSequence, Resonator, Square
from quchip.engine import build_problem


pytestmark = pytest.mark.unit


def _static_problem(
    backend: str,
    *,
    levels: int = 3,
    T1: float | None = None,
    points: int = 3,
    options: dict | None = None,
):
    qubit = DuffingTransmon(
        freq=5.0,
        anharmonicity=-0.25,
        levels=levels,
        label="q",
        T1=T1,
    )
    chip = Chip([qubit], frame="lab", backend=backend)
    problem = build_problem(
        chip,
        [],
        np.linspace(0.0, 1.0, points),
        options=options,
    )
    return chip.backend, problem


def _resolved_options(backend, problem) -> dict:
    prepared = backend.prepare_hamiltonian(problem.engine_result, problem.tlist)
    return backend._resolve_solve_config(problem, prepared)[3]


class TestQuTiPStaticPropagatorSelection:
    """QuTiP should diagonalize only eligible constant generators."""


    def test_explicit_diagonalization_rejects_adaptive_tolerances(self) -> None:
        """Explicit diagonal propagation rejects numerical settings it cannot apply."""
        backend, problem = _static_problem("qutip", options={"method": "diag", "atol": 1e-10})
        with pytest.raises(ValueError, match="diag.*options"):
            _resolved_options(backend, problem)

    @pytest.mark.parametrize("controls", [{"atol": 1e-10, "rtol": 1e-8}, {"nsteps": 17}, {"max_step": 0.01}])
    def test_explicit_controls_retain_adaptive_integration(self, controls: dict) -> None:
        """User-supplied adaptive controls prevent an incompatible automatic method change."""
        backend, problem = _static_problem("qutip", options=controls)
        options = _resolved_options(backend, problem)
        assert options.get("method") != "diag"
        for name, value in controls.items():
            assert options[name] == value

    def test_driven_problem_retains_adaptive_integrator(self) -> None:
        """Any explicit Hamiltonian time dependence should keep QuTiP's adaptive integrator."""
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(qubit, label="xy")
        chip = Chip([qubit], frame="rotating", backend="qutip")
        chip.wire(drive)
        sequence = QuantumSequence(chip)
        sequence.schedule(
            drive,
            envelope=Square(duration=1.0, amplitude=0.02),
            freq=chip.freq(qubit),
        )
        problem = sequence.build_problem(tlist=np.linspace(0.0, 1.0, 3))

        options = _resolved_options(chip.backend, problem)

        assert problem.engine_result.dynamic_terms
        assert "method" not in options


def test_network_generated_static_terms_keep_the_adaptive_integrator() -> None:
    """Cascade-generated static terms can defeat diagonalization, so the adaptive integrator stays."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=5.0, levels=2, label="b")
    network = PortNetwork(label="line")
    port_a = network.port("a_port", target=first, rate=0.05)
    port_b = network.port("b_port", target=second, rate=0.05)
    network.cascade(port_a, port_b)
    network.expose("feedline", input=port_a.input, output=port_b.output)
    chip = Chip([first, second], port_network=network, frame=5.0, backend="qutip")
    problem = build_problem(chip, [], np.linspace(0.0, 1.0, 3))
    assert problem.engine_result.slh.has_network_hamiltonian

    options = _resolved_options(chip.backend, problem)

    assert "method" not in options
