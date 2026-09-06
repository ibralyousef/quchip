"""Automatic constant-generator solver selection at the backend boundary."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import ChargeDrive, Chip, DuffingTransmon, PortNetwork, QuantumSequence, Resonator, Square
from quchip.engine import build_problem


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

    def test_small_static_closed_problem_selects_diagonalization(self) -> None:
        """A closed static Hilbert space through dimension 64 should select QuTiP's diagonal propagator."""
        backend, problem = _static_problem("qutip", levels=64)

        options = _resolved_options(backend, problem)

        assert options["method"] == "diag"
        assert "nsteps" not in options
        assert "max_step" not in options

    def test_small_static_open_problem_selects_liouvillian_diagonalization(self) -> None:
        """A static dissipative Hilbert space through dimension 32 (Liouvillian 1024) diagonalizes."""
        backend, problem = _static_problem("qutip", levels=32, T1=20.0)

        options = _resolved_options(backend, problem)

        assert options["method"] == "diag"
        assert "nsteps" not in options
        assert "max_step" not in options

    @pytest.mark.parametrize(
        ("levels", "T1"),
        [(65, None), (33, 20.0)],
        ids=["closed-above-limit", "open-above-limit"],
    )
    def test_large_static_problem_retains_adaptive_integrator(
        self,
        levels: int,
        T1: float | None,
    ) -> None:
        """Static generators above their dimension limit should retain QuTiP's adaptive default."""
        backend, problem = _static_problem("qutip", levels=levels, T1=T1)

        options = _resolved_options(backend, problem)

        assert "method" not in options
        assert "nsteps" in options

    def test_explicit_method_is_never_overridden(self) -> None:
        """An explicit QuTiP method should remain authoritative for an otherwise eligible problem."""
        backend, problem = _static_problem("qutip", options={"method": "adams"})

        options = _resolved_options(backend, problem)

        assert options["method"] == "adams"

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


@pytest.mark.optional_backend
def test_dynamiqs_static_problem_retains_adaptive_integrator() -> None:
    """quchip must not automatically select Dynamiqs' newer matrix-exponential method."""
    pytest.importorskip("dynamiqs")
    backend, problem = _static_problem("dynamiqs", levels=3, points=2)

    options = _resolved_options(backend, problem)

    assert "method" not in options
    assert "max_steps" in options


def test_open_transmon_resonator_chip_diagonalizes_its_liouvillian() -> None:
    """A 4 x 5 open chip made static by its frame takes the diagonal propagator instead of adaptive stepping."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q", T1=20.0)
    resonator = Resonator(freq=6.0, levels=5, label="r")
    chip = Chip([qubit, resonator], frame="rotating", backend="qutip")
    problem = build_problem(chip, [], np.linspace(0.0, 1.0, 3))
    assert not problem.engine_result.dynamic_terms

    options = _resolved_options(chip.backend, problem)

    assert options["method"] == "diag"


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
