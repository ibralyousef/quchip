"""Derived models retain the source representation and string-state meaning."""

import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator, eliminate


def _model(*, coupled):
    q = DuffingTransmon(freq=5, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7, levels=4, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.01)] if coupled else None, basis="eigen")
    chip.set_frame({"q": np.array(4.9), "r": np.array(6.9)})
    chip.set_state_order("r", "q", levels={"a": 0, "b": 1})
    return chip


@pytest.mark.parametrize(
    "derive", [lambda c: c.clone(), lambda c: c.with_params({}), lambda c: Chip.from_dict(c.to_dict())],
)
def test_derived_model_retains_string_states(derive):
    source = _model(coupled=True)
    derived = derive(source)
    source.frame["q"][...] = 4.5
    assert derived.frame["q"] == pytest.approx(4.9)
    expected = source.bare_state("ab")
    actual = derived.bare_state("ab")
    np.testing.assert_allclose(derived.backend.to_array(actual), source.backend.to_array(expected))
    derived.set_state_order("q", "r", levels={"a": 1, "b": 0})
    np.testing.assert_allclose(source.backend.to_array(source.bare_state("ab")), source.backend.to_array(expected))


def test_partition_preserves_retained_basis_and_relative_state_order():
    source = _model(coupled=False)
    source.set_frame(np.array(4.9))
    components = source.partition().components
    source.frame[...] = 4.5
    assert all(component.chip.frame == pytest.approx(4.9) for component in components)
    components[0].chip.frame[...] = 4.7
    assert components[1].chip.frame == pytest.approx(4.9)
    for component in components:
        derived = component.chip
        assert derived.basis == "eigen"
        label = component.labels[0]
        assert derived.resolve().dims == (3 if label == "q" else 4,)
        np.testing.assert_allclose(
            derived.backend.to_array(derived.bare_state("b")),
            derived.backend.to_array(derived.bare_state({label: 1})),
        )


def test_elimination_preserves_retained_basis_and_state_symbols():
    source = _model(coupled=True)
    derived = eliminate(source, "r").chip
    source.frame["q"][...] = 4.5
    assert derived.frame == {"q": 4.9}
    assert derived.basis == "eigen"
    assert derived.resolve().dims == (3,)
    np.testing.assert_allclose(
        derived.backend.to_array(derived.bare_state("b")),
        derived.backend.to_array(derived.bare_state({"q": 1})),
    )


@pytest.mark.parametrize("rebind", [False, True])
def test_sequence_derivations_have_independent_models(rebind):
    """Schedule cloning and no-op parameter rebinding cannot share mutable devices."""
    from quchip import QuantumSequence

    source = _model(coupled=True)
    seq = QuantumSequence(source)
    derived = seq.with_params({}) if rebind else seq.clone()
    derived._chip["q"].freq = 5.5
    assert source["q"].freq == 5.0
