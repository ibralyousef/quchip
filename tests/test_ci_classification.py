"""Verify that lightweight CI cannot hide executable changes or invalid diffs."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


CLASSIFIER = Path(__file__).resolve().parents[1] / ".github/actions/classify-pr/classify.sh"


class ChangeClassificationTests(unittest.TestCase):
    """Exercise the CI classifier against real commit pairs."""

    def setUp(self):
        """Create an isolated repository with a source file and baseline commit."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init", "-q")
        self.git("config", "user.name", "CI policy test")
        self.git("config", "user.email", "ci@example.invalid")
        self.write("quchip/model.py", "value = 1\n")
        self.base = self.commit()

    def git(self, *args):
        """Run Git only inside the isolated fixture repository."""
        return subprocess.check_output(["git", *args], cwd=self.root, text=True).strip()

    def write(self, name, content):
        """Write one fixture path, including any parent directories."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def commit(self):
        """Capture fixture edits as the comparison commit."""
        self.git("add", ".")
        self.git("commit", "-qm", "Fixture")
        return self.git("rev-parse", "HEAD")

    def classify(self, head, event="pull_request"):
        """Return classifier output without hiding failures."""
        return subprocess.run(
            ["bash", str(CLASSIFIER)], cwd=self.root, text=True, capture_output=True,
            env={**os.environ, "BASE_SHA": self.base, "HEAD_SHA": head, "EVENT_NAME": event},
        )

    def test_documentation_is_lightweight(self):
        """Documentation and release metadata require no physics test environment."""
        self.write("docs/guide.md", "Guide\n")
        self.write("RELEASE_NOTES_1.0.md", "Notes\n")
        result = self.classify(self.commit())
        self.assertEqual((result.returncode, result.stdout), (0, "full_ci=false\n"))

    def test_executable_markdown_and_ci_policy_require_tests(self):
        """Example notebooks, test resources, and CI policy cannot use the Markdown exemption."""
        for name in ("examples/demo.md", "tests/fixture.md", "tools/helper.md", ".github/rulesets/main.json"):
            with self.subTest(name=name):
                self.git("reset", "--hard", self.base)
                self.write(name, "Changed\n")
                result = self.classify(self.commit())
                self.assertEqual((result.returncode, result.stdout), (0, "full_ci=true\n"))

    def test_moving_source_into_docs_still_requires_tests(self):
        """Renaming a source file cannot disguise its removal as a docs-only edit."""
        (self.root / "docs").mkdir()
        self.git("mv", "quchip/model.py", "docs/model.md")
        result = self.classify(self.commit())
        self.assertEqual((result.returncode, result.stdout), (0, "full_ci=true\n"))

    def test_invalid_revision_fails_without_a_lightweight_result(self):
        """An unreadable diff fails instead of silently reporting no changed files."""
        result = self.classify("missing-revision")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("full_ci=false", result.stdout)

    def test_manual_runs_request_full_coverage(self):
        """Explicit runs bypass PR path classification and request the test lanes."""
        result = self.classify(self.base, event="workflow_dispatch")
        self.assertEqual((result.returncode, result.stdout), (0, "full_ci=true\n"))
