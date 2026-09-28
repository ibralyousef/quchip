"""Dynamiqs smoke coverage for Kerr-cavity operators."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.extended, pytest.mark.optional_backend]

pytest.importorskip("dynamiqs")
pytest.importorskip("jax")

from quchip.backend import set_default_backend  # noqa: E402
from quchip.control.drives_two_photon import TwoPhotonDrive  # noqa: E402
from quchip.control.signal import AnalyticSignal  # noqa: E402
from quchip.devices.kerr_cavity import KerrCavity  # noqa: E402
from quchip.engine.ir import Constant  # noqa: E402


@pytest.mark.validation
def test_kerr_cavity_and_two_photon_drive_build_with_dynamiqs_backend() -> None:
    """Operator products must use backend matrix multiplication, not elementwise multiplication."""
    set_default_backend("dynamiqs")

    cav = KerrCavity(freq=5.0, kerr=0.25, levels=5, label="cav")
    drive = TwoPhotonDrive(target=cav)

    hamiltonian = cav.hamiltonian()
    operator = drive.hamiltonian(cav, AnalyticSignal(Constant(1.0)))

    a = np.diag(np.sqrt(np.arange(1, 5)), 1)
    n = np.diag(np.arange(5))
    np.testing.assert_allclose(hamiltonian.matrix(), 5.0 * n - 0.25 * n @ (n - np.eye(5)))
    np.testing.assert_allclose(operator.matrix(t=0.0), a @ a + a.T @ a.T)
