"""Mutable model values invalidate new calculations without changing old ones."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, Exact
from quchip.declarative import DeviceModel, parameter, setting
from quchip.declarative.expr import PhysicsExpr
from quchip.engine import build_problem
from quchip.engine.ir import CanonicalOperator


class BufferedMode(DeviceModel):
    payload: Any = setting()

    def local_hamiltonian(self, op, p):
        return self.payload["nested"][0]


class CuratedMode(DeviceModel):
    freq: Any = parameter(default=5.0)
    shift: Any = parameter(default=0.0)
    tunable_param_names = ("freq",)

    def local_hamiltonian(self, op, p):
        return (p.freq + p.shift) * op.n


def test_declared_parameter_outside_tuning_surface_is_copied_and_invalidates_cache():
    source = CuratedMode(shift=np.array(0.0), label="q")
    chip = Chip([source], frame="lab")
    before = chip.resolve()
    cloned = chip.clone()
    source.shift[...] = 2.0
    assert cloned.freq("q") == pytest.approx(5.0)
    assert chip.freq("q") == pytest.approx(7.0)
    assert chip.resolve() is not before



def test_declared_parameter_outside_fit_selection_is_bindable():
    chip = Chip([CuratedMode(label="q")], frame="lab")
    changed = chip.with_params({"q.shift": 2.0})
    assert changed.freq("q") == pytest.approx(7.0)
    assert chip.freq("q") == pytest.approx(5.0)
    assert tuple(changed["q"].tunable_params()) == ("freq",)


def test_scalar_bindings_and_literals_capture_numpy_values():
    scalar = np.array(5.0)
    chip = Chip([DuffingTransmon(freq=scalar, anharmonicity=-0.2, label="q")], frame="lab")
    resolved = chip.resolve()
    literal = PhysicsExpr.literal(scalar)
    scalar[...] = 7.0
    assert resolved.authored.matrix()[1, 1] == pytest.approx(5.0)
    assert literal.args[0] == pytest.approx(5.0)
    assert chip.freq("q") == pytest.approx(7.0)


def test_clone_preserves_internal_payload_aliases_without_sharing_source():
    matrix = np.diag([0.0, 5.0])
    mode = BufferedMode(
        payload={"nested": [matrix], "alias": matrix, "metadata": ({7: matrix, "units": "GHz"},)}, label="q")
    derived = mode.copy()
    assert derived.payload["alias"] is derived.payload["nested"][0]
    assert derived.payload["metadata"][0][7] is derived.payload["alias"]
    derived.payload["alias"][1, 1] = 7.0
    assert mode.payload["alias"][1, 1] == 5.0
    assert derived.hamiltonian().matrix()[1, 1] == pytest.approx(7.0)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_nested_settings_are_independent_and_calculations_capture_values(backend):
    mode = BufferedMode(payload={"nested": [np.diag([0.0, 5.0])]}, label="q")
    chip = Chip([mode], frame="lab", approximation=Exact(), backend=backend)
    clone = chip.clone()
    clone["q"].payload["nested"][0][1, 1] = 7.0
    assert chip.freq("q") == pytest.approx(5.0)
    assert clone.freq("q") == pytest.approx(7.0)

    resolved = chip.resolve()
    problem = build_problem(chip, [], np.array([0.0, 1.0]))
    before_authored = np.asarray(problem.engine_result.authored.matrix()).copy()
    before_numeric = np.asarray(problem.engine_result.hamiltonian().matrix()).copy()
    mode.payload["nested"][0][1, 1] = 9.0

    np.testing.assert_array_equal(problem.engine_result.authored.matrix(), before_authored)
    np.testing.assert_array_equal(problem.engine_result.hamiltonian().matrix(), before_numeric)
    assert chip.resolve() is not resolved
    assert chip.freq("q") == pytest.approx(9.0)
    fresh = build_problem(chip, [], np.array([0.0, 1.0]))
    np.testing.assert_allclose(fresh.engine_result.hamiltonian().matrix(), before_numeric * 9 / 5)


@pytest.mark.parametrize("layout", ["dense", "csr", "dia"])
def test_canonical_operator_owns_all_numpy_buffers(layout):
    metadata = dict(dims=(2,), basis="native", subsystem_labels=("q",))
    if layout == "dense":
        buffers = {"values": np.diag([1.0, 2.0])}
        operator = CanonicalOperator.from_dense(**buffers, **metadata)
    elif layout == "csr":
        buffers = {"values": np.array([1.0, 2.0]), "indices": np.array([0, 1]), "indptr": np.array([0, 1, 2])}
        operator = CanonicalOperator.from_csr(**buffers, shape=(2, 2), **metadata)
    else:
        buffers = {"values": np.array([[1.0, 2.0]]), "offsets": np.array([0])}
        operator = CanonicalOperator.from_dia(**buffers, shape=(2, 2), **metadata)
    for name, value in buffers.items():
        expected = value.copy()
        value[...] = 0
        np.testing.assert_array_equal(getattr(operator, name), expected)
        assert not getattr(operator, name).flags.writeable


def test_copied_model_keeps_jax_gradient_through_nested_values():
    def frequency(value):
        mode = BufferedMode(payload={"nested": [jnp.diag(jnp.array([0.0, value]))]}, label="q")
        chip = Chip([mode], frame="lab", approximation=Exact(), backend="dynamiqs")
        return chip.clone().freq("q") ** 2

    assert jax.jit(jax.grad(frequency))(5.0) == pytest.approx(10.0)


def test_mutable_declaration_defaults_belong_to_each_instance():
    """Editing an instance cannot modify another instance's declared defaults."""
    class Defaults(DeviceModel):
        freq: Any = parameter(default=np.array(5.0))
        config: Any = setting(default={"channels": [0]})

        def local_hamiltonian(self, op, p):
            return p.freq * op.n

    first, second = Defaults(), Defaults()
    first.freq[...] = 7.0
    first.config["channels"].append(1)
    assert second.freq == 5.0
    assert second.config == {"channels": [0]}
