"""Contract coverage for the public chip-transformation guide."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import jupytext
import nbformat


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "02_reduce_and_replay.md"
EXAMPLE_IPYNB = ROOT / "examples" / "02_reduce_and_replay.ipynb"
RESULT_RE = re.compile(r"^RESULT reduction=(\{.*\})$", re.MULTILINE)


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


def test_executed_receipt_records_validity_and_forward_error() -> None:
    """The active-patch comparison records validity and a forward observable error."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])
    assert receipt["active_labels"] == ["q0", "q1"]
    assert receipt["eliminated_labels"] == ["q3", "q2"]
    assert receipt["full_dimension"] == 81 and receipt["reduced_dimension"] == 9
    assert receipt["all_folds_valid"] is True
    assert 0.03 < receipt["maximum_g_over_delta"] < 0.1
    assert receipt["same_schedule"] is True
    assert receipt["maximum_population_residual"] < receipt["residual_tolerance"]
    assert math.isfinite(receipt["maximum_population_residual"])
