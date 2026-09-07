"""Multi-device simulation and crosstalk integration tests.

Physics tests:
  1. Two resonant transmons with capacitive coupling — vacuum Rabi
     oscillation: P(|01⟩, t) = sin²(2π·g·t).
  2. Crosstalk structural: more Hamiltonian terms when crosstalks present.
  3. Crosstalk dynamics: leaked signal excites the victim device measurably.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quchip.backend.qutip import QuTiPBackend
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip import ControlEquipment
from quchip.control.signal import Crosstalk
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Square
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import DriveOp


class TestMultiDeviceSimulation:
    """Verify two-transmon population transfer via capacitive coupling.

    Physics: H = ω·n̂₀ + ω·n̂₁ + g·(a₀†a₁ + a₀a₁†)  (RWA).
    With ω₀ = ω₁ (resonant) and initial state |10⟩:
        P(|01⟩, t) = sin²(2π·g·t)
        P(|10⟩, t) = cos²(2π·g·t)

    g = 0.01 GHz → full oscillation period = 1/(2g) = 50 ns.
    """

    FREQ = 5.0  # GHz, same for both transmons (resonant)
    ANHARMONICITY = -0.25  # GHz
    LEVELS = 3  # Fock space truncation
    G = 0.01  # GHz, coupling strength
    DURATION = 50.0  # ns, one full oscillation
    N_POINTS = 501  # time grid density

    def _setup_and_run(self):
        """Build two-transmon chip and simulate free evolution from |10⟩."""
        q0 = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ANHARMONICITY,
            levels=self.LEVELS,
            label="q0",
        )
        q1 = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ANHARMONICITY,
            levels=self.LEVELS,
            label="q1",
        )
        coupling = Capacitive(q0, q1, g=self.G)
        chip = Chip([q0, q1], [coupling], frame="lab")

        psi0 = chip.bare_state(q0=1)

        tlist = np.linspace(0, self.DURATION, self.N_POINTS)

        result = simulate(
            chip,
            [],
            tlist,
            initial_state=psi0,
        )
        return tlist, result

    def test_vacuum_rabi_q0(self, backend: QuTiPBackend) -> None:
        """P(q0, level=1) matches cos²(2π·g·t) within 1%."""
        tlist, result = self._setup_and_run()

        p1_q0 = result.population("q0", level=1)
        p1_analytic = np.cos(2 * np.pi * self.G * tlist) ** 2

        npt.assert_allclose(
            p1_q0,
            p1_analytic,
            atol=0.01,
            err_msg=(
                f"q0 |1⟩ population deviates from cos²(2π·g·t). "
                f"Max deviation: {np.max(np.abs(p1_q0 - p1_analytic)):.4f}"
            ),
        )

    def test_vacuum_rabi_q1(self, backend: QuTiPBackend) -> None:
        """P(q1, level=1) matches sin²(2π·g·t) within 1%."""
        tlist, result = self._setup_and_run()

        p1_q1 = result.population("q1", level=1)
        p1_analytic = np.sin(2 * np.pi * self.G * tlist) ** 2

        npt.assert_allclose(
            p1_q1,
            p1_analytic,
            atol=0.01,
            err_msg=(
                f"q1 |1⟩ population deviates from sin²(2π·g·t). "
                f"Max deviation: {np.max(np.abs(p1_q1 - p1_analytic)):.4f}"
            ),
        )

class TestCrosstalkIntegration:
    """Verify crosstalk terms flow through the Hamiltonian pipeline."""

    FREQ_Q0 = 5.0
    FREQ_Q1 = 5.5  # off-resonant to isolate crosstalk effect
    ANHARMONICITY = -0.25
    LEVELS = 3
    OMEGA = 0.02  # drive amplitude (GHz), Ω/|α| = 0.08
    BETA = 0.1  # crosstalk coefficient
    DURATION = 50.0  # ns

    def _make_chip_and_drives(self):
        """Build two-transmon chip with drives on both devices."""
        q0 = DuffingTransmon(
            freq=self.FREQ_Q0,
            anharmonicity=self.ANHARMONICITY,
            levels=self.LEVELS,
            label="q0",
        )
        q1 = DuffingTransmon(
            freq=self.FREQ_Q1,
            anharmonicity=self.ANHARMONICITY,
            levels=self.LEVELS,
            label="q1",
        )
        drive_a = ChargeDrive(target=q0)
        drive_b = ChargeDrive(target=q1)
        chip = Chip([q0, q1], frame="lab")
        chip.connect(ControlEquipment(lines=[drive_a, drive_b]))
        return chip, q0, q1, drive_a, drive_b

    def test_dynamics_victim_excitation(self) -> None:
        """Crosstalk causes measurable excitation on the victim device."""
        chip_no_xt, q0_a, q1_a, drive_a_no, drive_b_no = self._make_chip_and_drives()
        chip_xt, q0_b, q1_b, drive_a_xt, drive_b_xt = self._make_chip_and_drives()

        envelope = Square(duration=self.DURATION, amplitude=self.OMEGA)
        tlist = np.linspace(0, self.DURATION, 501)

        drive_op_no = DriveOp(
            target_label="q0",
            envelope=envelope,
            freq=self.FREQ_Q0,
            drive_label=drive_a_no.label,
        )
        drive_op_xt = DriveOp(
            target_label="q0",
            envelope=envelope,
            freq=self.FREQ_Q0,
            drive_label=drive_a_xt.label,
        )

        result_no_xt = simulate(
            chip_no_xt,
            [drive_op_no],
            tlist,
        )

        src_key = drive_a_xt.label
        vic_key = drive_b_xt.label
        chip_xt.connect(ControlEquipment(
            lines=[drive_a_xt, drive_b_xt],
            signal_chain=[Crosstalk(source=src_key, victim=vic_key, beta=self.BETA)],
        ))
        result_with_xt = simulate(
            chip_xt,
            [drive_op_xt],
            tlist,
        )

        p1_no_xt = result_no_xt.population("q1", level=1)
        p1_with_xt = result_with_xt.population("q1", level=1)

        max_no_xt = np.max(p1_no_xt)
        max_with_xt = np.max(p1_with_xt)

        assert max_with_xt > max_no_xt, (
            f"Crosstalk should increase victim excitation: "
            f"max P(q1) with xt = {max_with_xt:.6f}, "
            f"without = {max_no_xt:.6f}"
        )

    def test_crosstalk_coefficient_scales_effect(self) -> None:
        """Larger beta produces larger victim excitation."""
        results_max_p1 = []

        for beta in [0.05, 0.2]:
            chip, q0, q1, drive_a, drive_b = self._make_chip_and_drives()
            envelope = Square(duration=self.DURATION, amplitude=self.OMEGA)
            tlist = np.linspace(0, self.DURATION, 501)

            drive_op = DriveOp(
                target_label="q0",
                envelope=envelope,
                freq=self.FREQ_Q0,
                drive_label=drive_a.label,
            )
            src_key = drive_a.label
            vic_key = drive_b.label
            chip.connect(ControlEquipment(
                lines=[drive_a, drive_b],
                signal_chain=[Crosstalk(source=src_key, victim=vic_key, beta=beta)],
            ))
            result = simulate(
                chip,
                [drive_op],
                tlist,
            )
            p1_q1 = result.population("q1", level=1)
            results_max_p1.append(np.max(p1_q1))

        assert results_max_p1[1] > results_max_p1[0], (
            f"Larger beta should produce more excitation: "
            f"beta=0.05 → {results_max_p1[0]:.6f}, "
            f"beta=0.2 → {results_max_p1[1]:.6f}"
        )
