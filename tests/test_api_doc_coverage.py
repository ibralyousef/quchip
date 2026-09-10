"""Verify that documentation coverage detects missing descriptions and drift."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import dataclass
from importlib.util import find_spec

if find_spec("sphinx") is None:
    raise unittest.SkipTest("API documentation checks require the docs extra")

from tools.check_api_docs import check_callable, documented_fields


class DocumentationCoverageTests(unittest.TestCase):
    """Exercise parsed docstrings and runtime signatures, including generated ones."""

    def test_prose_or_types_alone_do_not_count_as_descriptions(self):
        """A mentioned parameter or a type-only entry remains undocumented."""
        fields, _ = documented_fields(
            "Use duration in ns.\n\nParameters\n----------\nduration : float\n"
        )
        self.assertEqual(fields, set())

    def test_variadic_and_grouped_fields_are_recognized(self):
        """Escaped variadics and grouped attributes map to each real name."""
        params, attrs = documented_fields(
            "Parameters\n----------\n*axes\n    Independent axes.\n"
            "**options\n    Solver options.\n\nAttributes\n----------\n"
            "initial, final : float\n    Endpoint values.\n"
        )
        self.assertEqual(params, {"axes", "options"})
        self.assertEqual(attrs, {"initial", "final"})

    def test_runtime_signature_detects_added_and_obsolete_parameters(self):
        """Synthesized signatures are checked rather than the generic Python wrapper."""
        def model(**kwargs):
            """Parameters
            ----------
            old_name : float
                Retired argument.
            """
        model.__signature__ = inspect.Signature([
            inspect.Parameter("frequency", inspect.Parameter.KEYWORD_ONLY)
        ])
        issue = check_callable("model", model)
        self.assertEqual(issue["missing"], ["frequency"])
        self.assertEqual(issue["extra"], ["old_name"])

    def test_result_attributes_cover_public_fields(self):
        """Result attributes cover public dataclass fields without exposing storage internals."""
        @dataclass
        class Result:
            """Attributes
            ----------
            values : object
                Observable values on the saved time grid.
            """
            values: object
            _cache: object = None

        self.assertIsNone(check_callable("Result", Result))
