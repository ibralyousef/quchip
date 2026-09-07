"""Contract coverage for the public differentiability guide."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import jupytext
import nbformat


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "03_differentiate_a_driven_chip.md"
EXAMPLE_IPYNB = ROOT / "examples" / "03_differentiate_a_driven_chip.ipynb"
RESULT_RE = re.compile(r"^RESULT gradient=(\{.*\})$", re.MULTILINE)
EXPERIMENTAL_RESULT_RE = re.compile(
    r"^RESULT experimental_statics=(\{.*\})$",
    re.MULTILINE,
)


def _code(notebook: dict) -> str:
    return "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _stream_output(notebook: dict) -> str:
    return "".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_guide_code_matches_the_executed_notebook() -> None:
    """The reader-facing Markdown contains exactly the executed code."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    assert _code(authored) == _code(executed)
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")


def test_experimental_static_fit_recovers_fluxonium_parameters_on_holdout_data() -> None:
    """The experimental fit uses sparse training data and reports held-out agreement."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = EXPERIMENTAL_RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])
    assert receipt["fit_success"] is True
    assert receipt["training_points"] == 20
    assert receipt["holdout_points"] == 133
    assert max(abs(value) for value in receipt["relative_parameter_error"]) < 0.01
    assert receipt["holdout_median_absolute_error_mhz"] < 1.0


def test_executed_receipt_records_multi_sequence_gradients() -> None:
    """The receipt records pulse derivatives and the joint-loss shape."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])
    assert receipt["backend"] == "dynamiqs"
    assert receipt["solver"] == "sesolve"
    assert receipt["multi_sequence_count"] == 3
    assert receipt["multi_sequence_jacobian_shape"] == [3, 3]
    assert len(receipt["multi_sequence_loss_gradient"]) == 3
    assert all(math.isfinite(value) for value in receipt["multi_sequence_loss_gradient"])
    gradients = receipt["gradient_per_reference_perturbation"]
    assert gradients["pulse.0.amplitude"] > 0.0
    assert gradients["pulse.0.sigmas"] < 0.0
    assert gradients["pulse.0.freq"] > 0.0
