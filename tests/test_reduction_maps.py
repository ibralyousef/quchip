"""Captured reduction maps preserve retained-state norm and observable meaning."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Exact, Resonator, eliminate


def _model(backend, g=0.05):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    return Chip([q, r], [Capacitive(q, r, g=g)], backend=backend, approximation=Exact())


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("method", ["sw", "exact"])
def test_reduction_map_captures_lab_coordinates_and_preserves_expectations(backend, method):
    source = _model(backend)
    result = eliminate(source, "r", method=method)
    assert result.chip["q"].freq == source["q"].freq
    mapping = result.mapping
    assert mapping.source_labels == ("q", "r")
    assert mapping.target_labels == ("q",)
    assert mapping.source_dims == (3, 3) and mapping.target_dims == (3,)
    # Mutation before first access must not change a deferred map.
    reference = eliminate(_model(backend), "r", method=method).mapping.embedding
    source["q"].freq = 5.7
    result.chip["q"].freq = 4.6
    np.testing.assert_allclose(mapping.embedding, reference, atol=1e-12)
    np.testing.assert_allclose(mapping.embedding.conj().T @ mapping.embedding, np.eye(3), atol=1e-12)
    ket = np.array([1., 1j, .5])[:, None] / 1.5
    operator = np.diag(np.arange(9)) + .2j * (np.eye(9, k=1) - np.eye(9, k=-1))
    lifted = source.backend.to_array(mapping.lift_state(ket))
    projected = source.backend.to_array(mapping.project_state(lifted))
    observable = source.backend.to_array(mapping.project_operator(operator))
    np.testing.assert_allclose(projected, ket, atol=1e-12)
    np.testing.assert_allclose(lifted.conj().T @ operator @ lifted, ket.conj().T @ observable @ ket, atol=1e-12)
    rho = ket @ ket.conj().T
    lifted_rho = source.backend.to_array(mapping.lift_state(rho))
    np.testing.assert_allclose(lifted_rho, lifted @ lifted.conj().T, atol=1e-12)
    np.testing.assert_allclose(source.backend.to_array(mapping.project_state(lifted_rho)), rho, atol=1e-12)
    rejected = (np.eye(9) - reference @ reference.conj().T)[:, :1]
    np.testing.assert_allclose(source.backend.to_array(mapping.project_state(rejected)), 0., atol=1e-12)
    discarded = rejected / np.linalg.norm(rejected)
    mixture = .4 * lifted_rho + .6 * discarded @ discarded.conj().T
    retained = source.backend.to_array(mapping.project_state(mixture))
    assert np.trace(retained).real == pytest.approx(.4, abs=1e-12)
    with pytest.raises(ValueError, match="shape"):
        mapping.project_operator(np.eye(3))
    with pytest.raises(ValueError, match="shape"):
        mapping.lift_state(np.zeros((9, 1)))


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_reduction_map_gradient_matches_finite_difference(method):
    def population(g):
        model = _model("dynamiqs", g)
        mapping = eliminate(model, "r", method=method).mapping
        lifted = model.backend.to_array(mapping.lift_state(jnp.array([0., 1., 0.])))
        return jnp.real(jnp.abs(lifted[1, 0]) ** 2)

    g, step = .05, 1e-5
    expected = (population(g + step) - population(g - step)) / (2 * step)
    assert jax.jit(jax.grad(population))(g) == pytest.approx(float(expected), rel=1e-6, abs=1e-10)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("basis", ["native", "eigen"])
def test_exact_map_transforms_non_diagonal_and_projected_local_bases(backend, basis):
    from quchip.declarative import DeviceModel, Scalar, parameter

    class TiltedMode(DeviceModel):
        freq: Scalar = parameter(default=5.)

        def local_hamiltonian(self, op, p):
            return p.freq * op.n + .3 * op.sigma_x + .2 * op.sigma_y

    q = TiltedMode(levels=3, label="q")
    q.projection_levels = 2
    r = Resonator(freq=7., levels=3, label="r")
    source = Chip([q, r], [Capacitive(q, r, g=.05)], basis=basis, backend=backend, approximation=Exact())
    result = eliminate(source, "r", method="exact")
    mapping = result.mapping
    source_h = source.resolve(frame="lab").hamiltonian().matrix(backend=source.backend)
    target_h = result.chip.resolve(frame="lab").hamiltonian().matrix(backend=source.backend)
    np.testing.assert_allclose(source.backend.to_array(mapping.project_operator(source_h)), target_h, atol=1e-11)
    size = mapping.target_dims[0]
    np.testing.assert_allclose(mapping.embedding.conj().T @ mapping.embedding, np.eye(size), atol=1e-12)
    assert size == (3 if basis == "native" else 2)


def test_exact_map_reproduces_retained_closed_dynamics():
    from scipy.linalg import expm

    source = _model("qutip")
    result = eliminate(source, "r", method="exact")
    target_state = result.chip.bare_state(q=1)
    lifted = source.backend.to_array(result.mapping.lift_state(target_state))
    h = np.asarray(source.resolve(frame="lab").hamiltonian().matrix(backend=source.backend))
    h_reduced = np.asarray(result.chip.resolve(frame="lab").hamiltonian().matrix(backend=source.backend))
    # Hamiltonians returned by the public inspection path are in GHz.
    for time in (0., .4, 2.7):
        full_evolved = expm(-2j * np.pi * h * time) @ lifted
        reduced_evolved = expm(-2j * np.pi * h_reduced * time) @ source.backend.to_array(target_state)
        expected = source.backend.to_array(result.mapping.lift_state(reduced_evolved))
        np.testing.assert_allclose(full_evolved, expected, atol=1e-11)


def test_sw_map_preserves_norm_while_dynamics_converge_perturbatively():
    from scipy.linalg import expm

    errors = []
    h_errors = []
    for g in (.1, .05, .025):
        source = _model("qutip", g)
        result = eliminate(source, "r")
        b = np.asarray(result.mapping.embedding)
        h = np.asarray(source.resolve(frame="lab").hamiltonian().matrix())
        hr = np.asarray(result.chip.resolve(frame="lab").hamiltonian().matrix())
        ket = np.array([0., 1., 0.])
        errors.append(np.linalg.norm(expm(-2j * np.pi * h * .3) @ b @ ket - b @ expm(-2j * np.pi * hr * .3) @ ket))
        h_errors.append(np.linalg.norm(b.conj().T @ h @ b - hr))
        np.testing.assert_allclose(b.conj().T @ b, np.eye(3), atol=1e-12)
    # This parity-symmetric model has fourth-order projected-H error;
    # its omitted subspace leakage already affects full dynamics at second order.
    assert np.all(np.asarray(errors[:-1]) / errors[1:] > 3.8)
    assert np.all(np.asarray(h_errors[:-1]) / h_errors[1:] > 15.)
