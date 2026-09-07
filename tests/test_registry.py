"""Registration and payload dispatch for serializable component families."""

from __future__ import annotations

import abc

import pytest

from quchip.utils.registry import Registrable


def test_concrete_subclass_round_trips_through_registry_root():
    """Only concrete subclasses register, under their fully qualified names."""
    class Root(Registrable, registry_root=True):
        pass

    class AbstractMid(Root, abc.ABC):
        @abc.abstractmethod
        def value(self) -> int: ...

    class Concrete(AbstractMid):
        def value(self) -> int:
            return 7

    key = f"{Concrete.__module__}.{Concrete.__qualname__}"
    assert Root._registry == {key: Concrete}
    assert Concrete().to_dict()["type"] == key
    assert Root.from_dict({"type": key}).value() == 7


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
