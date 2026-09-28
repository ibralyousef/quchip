"""Registration and payload dispatch for serializable component families."""

from __future__ import annotations


import pytest

from quchip.utils.registry import Registrable


pytestmark = pytest.mark.unit


def test_unknown_type_lists_registered_choices():
    """An unknown type reports the available registered classes."""
    class Root(Registrable, registry_root=True):
        pass

    class Zeta(Root):
        pass

    class Alpha(Root):
        pass

    with pytest.raises(ValueError) as exc_info:
        Root.from_dict({"type": "nonexistent.Type"})
    message = str(exc_info.value)
    assert Alpha._type_key() in message
    assert Zeta._type_key() in message
