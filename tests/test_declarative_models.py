from __future__ import annotations

import inspect

import jax
import numpy as np
import numpy.testing as npt
import pytest

from quchip import CouplingModel, DeviceModel, Envelope, Scalar, qnp, parameter
from quchip.declarative import setting
from quchip.devices.base import BaseDevice


class CosineEnvelope(Envelope):
    duration: Scalar = parameter(positive=True)
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t):
        return self.amplitude * (1.0 - qnp.cos(qnp.pi * t / self.duration))


def test_custom_envelope_samples_without_xp_argument():
    """A custom Envelope subclass samples correctly through the base pipeline without an explicit xp argument."""
    env = CosineEnvelope(duration=10.0, amplitude=2.0)
    samples = env.sample(np.asarray([0.0, 5.0, 10.0]))
    npt.assert_allclose(np.asarray(samples), np.asarray([0.0, 2.0, 4.0]), atol=1e-7)


class HarmonicMode(DeviceModel):
    freq: Scalar = parameter(positive=True)
    approximation = None

    def local_hamiltonian(self, op, p):
        return p.freq * op.n


class ConfiguredMode(DeviceModel):
    freq: Scalar = parameter(unit="GHz")
    basis_name: str = setting()

    def local_hamiltonian(self, op, p):
        return p.freq * op.n


def test_device_setting_is_keyword_only_and_round_trips():
    with pytest.raises(TypeError, match="basis_name"):
        ConfiguredMode(5.0)
    signature = inspect.signature(ConfiguredMode)
    assert signature.parameters["basis_name"].kind is inspect.Parameter.KEYWORD_ONLY

    mode = ConfiguredMode(5.0, basis_name="eigen", levels=3, label="m")
    restored = BaseDevice.from_dict(mode.to_dict())

    assert isinstance(restored, ConfiguredMode)
    assert restored.basis_name == "eigen"


def test_device_setting_is_jax_structural_data():
    mode = ConfiguredMode(5.0, basis_name="eigen", levels=3, label="m")

    leaves = jax.tree_util.tree_leaves(mode)

    assert 5.0 in leaves
    assert "eigen" not in leaves


def test_tunable_param_names_derived_default_covers_all_declared_fields():
    """A DeviceModel subclass with no explicit tunable_param_names exposes exactly its declared parameter() fields."""
    class _DerivedTunables(DeviceModel):
        freq: Scalar = parameter(positive=True)
        anharm: Scalar = parameter()

        def local_hamiltonian(self, op, p):
            return p.freq * op.n

    dev = _DerivedTunables(freq=5.0, anharm=-0.2, levels=3)
    assert dev.tunable_param_names == ("freq", "anharm")
    assert set(dev.tunable_params()) == {"freq", "anharm"}


def test_tunable_param_names_explicit_curation_is_exact():
    """An explicit tunable_param_names tuple exposes exactly those names, excluding other declared fields."""
    class _CuratedTunables(DeviceModel):
        freq: Scalar = parameter(positive=True)
        quality_factor: Scalar = parameter(default=None)
        tunable_param_names = ("freq",)

        def local_hamiltonian(self, op, p):
            return p.freq * op.n

    dev = _CuratedTunables(freq=5.0, levels=3)
    assert set(dev.tunable_params()) == {"freq"}


def test_tunable_param_names_explicit_empty_freezes_device():
    """An explicit empty tunable_param_names tuple exposes no tunable parameters (deliberate inverse-design freeze)."""
    class _FrozenTunables(DeviceModel):
        freq: Scalar = parameter(positive=True)
        tunable_param_names = ()

        def local_hamiltonian(self, op, p):
            return p.freq * op.n

    dev = _FrozenTunables(freq=5.0, levels=3)
    assert dev.tunable_params() == {}


def test_tunable_param_names_inherited_explicit_curation_is_not_re_derived():
    """A subclass of an explicitly-curated DeviceModel inherits that exact curation instead of re-deriving."""
    class _CuratedParent(DeviceModel):
        freq: Scalar = parameter(positive=True)
        quality_factor: Scalar = parameter(default=None)
        tunable_param_names = ()

        def local_hamiltonian(self, op, p):
            return p.freq * op.n

    class _CuratedChild(_CuratedParent):
        extra: Scalar = parameter(default=1.0)

    dev = _CuratedChild(freq=5.0, levels=3)
    assert dev.tunable_param_names == ()
    assert dev.tunable_params() == {}


def test_tunable_param_names_derived_lineage_re_derives_with_new_fields():
    """A subclass of a purely derived-default DeviceModel re-derives to include its own new declared fields."""
    class _DerivedParent(DeviceModel):
        a: Scalar = parameter(positive=True)
        b: Scalar = parameter(default=0.0)

        def local_hamiltonian(self, op, p):
            return p.a * op.n

    class _DerivedChild(_DerivedParent):
        c: Scalar = parameter(default=0.0)

    assert _DerivedParent.tunable_param_names == ("a", "b")
    assert _DerivedChild.tunable_param_names == ("a", "b", "c")


def test_tunable_param_names_accepts_a_plain_class_attribute():
    """An explicit tunable_param_names entry may name a genuine class attribute, not only a parameter() field."""
    class _WithClassAttr(DeviceModel):
        freq: Scalar = parameter(positive=True)
        derived_freq = 0.0
        tunable_param_names = ("freq", "derived_freq")

        def local_hamiltonian(self, op, p):
            return p.freq * op.n

    dev = _WithClassAttr(freq=5.0, levels=3)
    assert set(dev.tunable_param_names) == {"freq", "derived_freq"}


@pytest.mark.parametrize(
    "names,error,message",
    [
        (("not_a_field",), ValueError, "not a declared parameter"),
        (("freq", "freq"), ValueError, "duplicate"),
        ("freq", TypeError, "must be a tuple"),
        ((1,), TypeError, "must be strings"),
    ],
)
def test_invalid_tunable_declarations_fail_at_class_definition(names, error, message):
    """Reject unknown, duplicate, and malformed tunable-field declarations."""
    with pytest.raises(error, match=message):
        class BadTunables(HarmonicMode):
            tunable_param_names = names


class NumberNumber(CouplingModel):
    chi: Scalar = parameter()

    def interaction(self, a, b, p):
        return p.chi * a.n * b.n


def test_custom_coupling_compiles_without_backend_tensor_calls():
    """A custom CouplingModel subclass compiles its interaction to an operator on the joint two-device Hilbert space."""
    a = HarmonicMode(freq=5.0, levels=3, label="a")
    b = HarmonicMode(freq=6.0, levels=4, label="b")
    coupling = NumberNumber(a, b, chi=0.01)
    h = coupling.interaction_hamiltonian().matrix()
    npt.assert_allclose(h, 0.01 * np.kron(np.diag(np.arange(3)), np.diag(np.arange(4))), atol=1e-12)


def test_time_terms_reject_values_outside_the_public_contract():
    """The private bridge rejects values outside the public time-term contract."""
    class BadDynamic(CouplingModel):
        g: Scalar = parameter()

        def interaction(self, a, b, p):
            return p.g * a.x * b.x

        def time_terms(self, a, b, p):
            return p.g * a.x * b.x

    a = HarmonicMode(freq=5.0, levels=3, label="a")
    b = HarmonicMode(freq=6.0, levels=4, label="b")
    coupling = BadDynamic(a, b, g=0.01)
    with pytest.raises(TypeError, match="must return TimeDependentTerm"):
        coupling._time_terms()


def test_tunable_capacitive_without_modulation_has_no_dynamic_term():
    """A purely static TunableCapacitive emits no dynamic interaction term."""
    from quchip import DuffingTransmon, TunableCapacitive

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.05, anharmonicity=-0.25, levels=3, label="q1")
    c = TunableCapacitive(q0, q1, g_0=0.02)
    assert c._time_terms() == ()


def test_declarative_device_round_trip_uses_declared_parameters():
    """A DeviceModel round-trips through to_dict()/from_dict() with its type and declared parameter values preserved."""
    mode = HarmonicMode(freq=7.0, levels=4, label="m")
    restored = BaseDevice.from_dict(mode.to_dict())
    assert isinstance(restored, HarmonicMode)
    assert restored.freq == 7.0
    assert restored.levels == 4
    assert restored.label == "m"
    npt.assert_allclose(restored.hamiltonian().matrix(), np.diag([0.0, 7.0, 14.0, 21.0]), atol=1e-12)


def test_declarative_envelope_round_trip_uses_declared_parameters():
    """An Envelope round-trips through to_dict()/from_dict() with type and declared parameters preserved."""
    env = CosineEnvelope(duration=10.0, amplitude=2.0)
    restored = Envelope.from_dict(env.to_dict())
    assert isinstance(restored, CosineEnvelope)
    assert restored.duration == 10.0
    assert restored.amplitude == 2.0
