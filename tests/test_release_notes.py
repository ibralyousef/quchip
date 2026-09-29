"""Check release-body extraction before the tag workflow publishes anything."""

import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/release_notes.py"


def extract(tmp_path, content, version="1.2.3"):
    """Run the same command as the release workflow against a temporary changelog."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(content)
    return subprocess.run(
        [sys.executable, str(SCRIPT), version, "--changelog", str(changelog)],
        capture_output=True, text=True,
    )


def test_version_section_preserves_links_without_neighboring_changes(tmp_path):
    """Only the requested release is emitted, with usable inline and reference links."""
    result = extract(tmp_path, """# Changelog

## Unreleased
- Future work.
## [1.2.3] - 2026-09-29
### Fixes
- Corrected the calculation. [#7]
- See [extensions](docs/extensions.md#custom-devices).
## [1.2.2] - 2026-09-28
- Earlier work.

[#7]: https://github.com/quchip/quchip/pull/7
""")
    assert result.returncode == 0, result.stderr
    assert "Corrected the calculation." in result.stdout
    assert "### Fixes" in result.stdout
    assert "Future work" not in result.stdout and "Earlier work" not in result.stdout
    assert "https://github.com/quchip/quchip/blob/v1.2.3/docs/extensions.md#custom-devices" in result.stdout
    assert "[#7]: https://github.com/quchip/quchip/pull/7" in result.stdout


@pytest.mark.parametrize("content", [
    "## [1.2.30]\n- Wrong version.\n",
    "## [1.2.3]\n\n## [1.2.2]\n- Previous release.\n",
    "## [1.2.3]\n\n[#7]: https://example.org\n",
    "## [1.2.3]\n- First copy.\n## [1.2.3]\n- Second copy.\n",
])
def test_invalid_release_fails_without_a_body(tmp_path, content):
    """Missing, empty, or duplicate releases cannot produce a publishable body."""
    result = extract(tmp_path, content)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "changelog section" in result.stderr


def test_last_release_has_one_copy_of_reference_definitions(tmp_path):
    """The final section keeps its links without duplicating trailing definitions."""
    result = extract(tmp_path, "## [1.2.3]\n- Fixed it. [#7]\n\n[#7]: https://example.org\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[#7]:") == 1


@pytest.mark.parametrize("version", ["0.2.0", "0.2.1", "0.3.0", "0.3.1", "0.3.2"])
def test_migrated_release_remains_available(version):
    """Every deleted release-body version remains extractable from the changelog."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), version], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip()) > 100
