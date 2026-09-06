"""Backend selection coverage for lazy dynamiqs resolution and error handling."""

from __future__ import annotations

import importlib
import re

import pytest

import quchip.backend as backend_module
from quchip.chip.chip import Chip


INSTALL_HINT = "DynamiqsBackend requires dynamiqs and JAX. Install with: pip install quchip[dynamiqs]"


def test_unknown_backend_name_still_raises_value_error() -> None:
    """Unknown backend names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown backend 'nope'"):
        backend_module.set_default_backend("nope")


@pytest.mark.parametrize("select", [backend_module.set_default_backend, lambda name: Chip([], backend=name)])
def test_missing_dynamiqs_extra_raises_install_hint(monkeypatch: pytest.MonkeyPatch, select) -> None:
    """Selecting dynamiqs without the extra installed raises the install hint."""
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "quchip.backend.dynamiqs":
            raise ImportError("dynamiqs unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=re.escape(INSTALL_HINT)):
        select("dynamiqs")
