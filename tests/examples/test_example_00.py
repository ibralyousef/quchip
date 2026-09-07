"""Contract coverage for the public dynamics and readout guide."""

from __future__ import annotations

import json
import re
from pathlib import Path

import jupytext
import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "00_hello_chip.md"
EXAMPLE_IPYNB = ROOT / "examples" / "00_hello_chip.ipynb"
RESULT_RE = re.compile(r"^RESULT (drive|readout)=(\{.*\})$", re.MULTILINE)


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


def test_executed_receipts_record_leakage_and_readout() -> None:
    """The executed notebook records the intended physical comparisons."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    receipts = {name: json.loads(payload) for name, payload in RESULT_RE.findall(_stream_output(executed))}
    assert receipts.keys() == {"drive", "readout"}
    drive = receipts["drive"]
    assert drive["final_p1"]["long"] > 0.90
    assert drive["peak_p2"]["short"] > 0.02
    assert drive["peak_p2"]["long"] < 0.25 * drive["peak_p2"]["short"]
    readout = receipts["readout"]
    assert readout["solver"] == "mesolve"
    assert readout["final_iq_separation"] > 0.01
    assert np.isfinite(readout["conditional_resonator_frequencies_ghz"]).all()
