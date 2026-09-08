"""The noisy resonator guide executes through the public measurement API."""

import contextlib
from pathlib import Path

import jupytext
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.examples
def test_noisy_vna_guide_reuses_physical_capture_for_both_receivers() -> None:
    """The public example retains source budgets and the predicted integration-time scaling."""
    markdown = jupytext.read(ROOT / "examples/04_noisy_vna_measurement.md")
    notebook = jupytext.read(ROOT / "examples/04_noisy_vna_measurement.ipynb")
    code = [cell.source for cell in markdown.cells if cell.cell_type == "code"]
    assert code == [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    namespace = {"__name__": "__noisy_vna_example__"}
    with contextlib.chdir(ROOT / "examples"):
        for source in code:
            exec(compile(source, "04_noisy_vna_measurement.md", "exec"), namespace)
    short_error = np.linalg.norm(namespace["short_ratio"] - namespace["ideal"])
    long_error = np.linalg.norm(namespace["long_ratio"] - namespace["ideal"])
    # HEMT white noise dominates; colored thermal/device corrections are below 0.01%.
    np.testing.assert_allclose(short_error/long_error, np.sqrt(10), rtol=1e-4)
    stats = namespace["calibrated"]
    np.testing.assert_allclose(sum(namespace["budget"].values()), stats.iq_covariance, atol=1e-12)
