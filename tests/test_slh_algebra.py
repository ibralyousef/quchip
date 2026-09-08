"""Public SLH algebra reproduces the Gough-James series, concatenation, and feedback rules."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from quchip.engine import concatenate, feedback_reduce, series_product
from quchip.engine.ir import CanonicalOperator, CollapseTerm, HamiltonianProgram, ResolvedSLH, SLHChannel, StaticTerm

_DIM = 3


def _operator(matrix: np.ndarray, tag: str) -> CanonicalOperator:
    return CanonicalOperator.from_dense(
        np.asarray(matrix, dtype=complex), dims=(_DIM,), basis="fock", subsystem_labels=("m",), tag=tag
    )


def _lowering() -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, _DIM)), 1).astype(complex)


def _system(
    label: str,
    scattering: np.ndarray,
    couplings: list[np.ndarray],
    hamiltonian: np.ndarray | None,
    *,
    dims: tuple[int, ...] = (_DIM,),
    labels: tuple[str, ...] = ("m",),
    carrier: float = 5.0,
    accessibility: str = "exposed",
) -> ResolvedSLH:
    channels = []
    for index, coupling in enumerate(couplings):
        key = f"{label}.{index}"
        operator = CanonicalOperator.from_dense(
            np.asarray(coupling, dtype=complex), dims=dims, basis="fock", subsystem_labels=labels, tag=key
        )
        collapse = CollapseTerm(operator=operator, rate=1.0, source=key, channel="port", frame_frequency=carrier)
        channels.append(
            SLHChannel(key=key, accessibility=accessibility, collapse=collapse, coupling_operator=operator)  # type: ignore[arg-type]
        )
    static = () if hamiltonian is None else (StaticTerm(operator=_operator(hamiltonian, f"{label}.H")),)
    return ResolvedSLH(
        scattering=np.asarray(scattering, dtype=complex),
        hamiltonian=HamiltonianProgram(static_terms=static),
        channels=tuple(channels),
    )


def _dense_h(slh: ResolvedSLH) -> np.ndarray:
    total = np.zeros((_DIM, _DIM), dtype=complex)
    for term in slh.H.static_terms:
        total = total + term.coefficient * np.asarray(term.operator.to_dense())
    return total


def _dense_l(slh: ResolvedSLH) -> list[np.ndarray]:
    return [np.asarray(op.to_dense()) for op in slh.L]


def _dag(matrix: np.ndarray) -> np.ndarray:
    return matrix.conj().T


def test_series_product_matches_textbook_rule() -> None:
    """G1 ▷ G2 = (S2 S1, L2 + S2 L1, H1 + H2 + (L2† S2 L1 − h.c.) / 2i)."""
    a = _lowering()
    number = _dag(a) @ a
    s1 = np.array([[0.0, 1.0], [1.0, 0.0]])
    s2 = np.array([[np.exp(0.3j), 0.0], [0.0, 1.0]])
    l1 = [0.2 * a, 0.1 * number]
    l2 = [0.3 * a, 0.05 * a @ a]
    first = _system("g1", s1, l1, 0.7 * number)
    second = _system("g2", s2, l2, 0.2 * a @ a + 0.2 * _dag(a @ a))

    combined = series_product(first, second)

    np.testing.assert_allclose(combined.S, s2 @ s1)
    expected_l = [l2[i] + sum(s2[i, j] * l1[j] for j in range(2)) for i in range(2)]
    for got, want in zip(_dense_l(combined), expected_l, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-12)
    cross = sum(_dag(l2[i]) @ (s2[i, j] * l1[j]) for i in range(2) for j in range(2))
    expected_h = _dense_h(first) + _dense_h(second) + (cross - _dag(cross)) / 2j
    np.testing.assert_allclose(_dense_h(combined), expected_h, atol=1e-12)
    assert [channel.key for channel in combined.channels] == ["g2.0", "g2.1"]


def test_series_product_rejects_channel_count_mismatch() -> None:
    """Series composition needs as many outputs on the first system as inputs on the second."""
    a = _lowering()
    first = _system("g1", np.eye(1), [a], None)
    second = _system("g2", np.eye(2), [a, a @ a], None)
    with pytest.raises(ValueError, match="channel"):
        series_product(first, second)


def test_concatenate_is_block_diagonal_and_prefixes_keys() -> None:
    """G1 ⊞ G2 stacks channels block-diagonally, adds Hamiltonians, and disambiguates keys on request."""
    a = _lowering()
    g = _system("g", np.array([[np.exp(0.5j)]]), [0.4 * a], 0.3 * _dag(a) @ a)

    with pytest.raises(ValueError, match="prefixes"):
        concatenate(g, g)
    combined = concatenate(g, g, prefixes=("left", "right"))

    np.testing.assert_allclose(combined.S, np.diag([np.exp(0.5j), np.exp(0.5j)]))
    assert [channel.key for channel in combined.channels] == ["left/g.0", "right/g.0"]
    np.testing.assert_allclose(_dense_h(combined), 2 * _dense_h(g))
    assert not combined.feeds(0, 1) and combined.feeds(1, 1)


def test_feedback_reduce_matches_textbook_rule() -> None:
    """Closing output x into input y follows the Gough-James reduction formulas."""
    a = _lowering()
    theta = 0.4
    s = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, np.exp(0.2j)]])
    ops = [0.2 * a, 0.1 * a @ a, 0.3 * _dag(a) @ a]
    h = 0.5 * _dag(a) @ a
    system = _system("g", s, ops, h)
    x, y = 1, 0
    keep_rows = [0, 2]
    keep_cols = [1, 2]

    reduced = feedback_reduce(system, output="g.1", input="g.0")

    gain = 1.0 / (1.0 - s[x, y])
    expected_s = s[np.ix_(keep_rows, keep_cols)] + np.outer(s[keep_rows, y], s[x, keep_cols]) * gain
    np.testing.assert_allclose(reduced.S, expected_s, atol=1e-12)
    expected_l = [ops[r] + s[r, y] * gain * ops[x] for r in keep_rows]
    for got, want in zip(_dense_l(reduced), expected_l, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-12)
    coupling = sum(_dag(ops[j]) * s[j, y] for j in range(3)) @ (gain * ops[x])
    np.testing.assert_allclose(_dense_h(reduced), h + (coupling - _dag(coupling)) / 2j, atol=1e-12)
    assert [channel.key for channel in reduced.channels] == ["g.1->g.0", "g.2"]


def test_feedback_reduce_rejects_singular_loop() -> None:
    """A loop with unit round-trip scattering has no instantaneous solution."""
    a = _lowering()
    system = _system("g", np.eye(2), [a, a @ a], None)
    with pytest.raises(ValueError, match="singular"):
        feedback_reduce(system, output="g.0", input="g.0")


def test_algebra_requires_common_hilbert_space() -> None:
    """Operators on different subsystem layouts cannot be combined."""
    a = _lowering()
    other = _system("o", np.eye(1), [np.eye(2)], None, dims=(2,), labels=("q",))
    with pytest.raises(ValueError, match="Hilbert"):
        series_product(_system("g", np.eye(1), [a], None), other)


def test_graph_cascade_equals_feedback_of_concatenated_ports() -> None:
    """Closing port a's output into port b's input reproduces the compiled cascade network."""
    from quchip import Chip, PortNetwork, Resonator

    def _chip(cascaded: bool) -> Chip:
        first = Resonator(freq=5.0, levels=3, label="a")
        second = Resonator(freq=5.0, levels=3, label="b")
        network = PortNetwork(label="line")
        port_a = network.port("pa", target=first, rate=0.04)
        port_b = network.port("pb", target=second, rate=0.09)
        if cascaded:
            network.cascade(port_a, port_b)
            network.expose("through", input=port_a.input, output=port_b.output)
        return Chip([first, second], port_network=network)

    open_system = _chip(False).resolve().slh
    closed = feedback_reduce(open_system, output="pa", input="pb")
    compiled = _chip(True).resolve().slh

    np.testing.assert_allclose(closed.S, compiled.S)
    np.testing.assert_allclose(closed.L[0].to_dense(), compiled.L[0].to_dense(), atol=1e-12)
    network_terms = [term for term in closed.H.static_terms if term.origin == "network"]
    compiled_terms = [term for term in compiled.H.static_terms if term.origin == "network"]
    assert len(network_terms) == len(compiled_terms) == 1
    np.testing.assert_allclose(network_terms[0].operator.to_dense(), compiled_terms[0].operator.to_dense(), atol=1e-12)
    assert [channel.key for channel in closed.channels] == ["pa->pb"]


def test_feedback_gain_traces_under_jit_and_grad() -> None:
    """A traced loop phase flows through feedback_reduce without concretization."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    a = _lowering()

    def reflected_gain(phase: Any) -> Any:
        theta = 0.3
        s = jnp.array(
            [[jnp.cos(theta) * jnp.exp(1j * phase), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]],
            dtype=complex,
        )
        traced = replace(_system("g", np.eye(2), [0.2 * a, 0.1 * a @ a], None), scattering=s)
        reduced = feedback_reduce(traced, output="g.1", input="g.1")
        return jnp.abs(reduced.S[0, 0]) ** 2

    value = jax.jit(reflected_gain)(0.4)
    gradient = jax.grad(reflected_gain)(0.4)
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))


def test_joins_require_matching_accessibility_and_carrier() -> None:
    """Joined legs must agree on visibility and carrier before their metadata can be merged."""
    a = _lowering()
    exposed = _system("g", np.eye(1), [a], None)
    hidden = _system("h", np.eye(1), [a], None, accessibility="hidden")
    detuned = _system("d", np.eye(1), [a], None, carrier=5.5)
    with pytest.raises(ValueError, match="accessibility"):
        series_product(exposed, hidden)
    with pytest.raises(ValueError, match="carrier"):
        series_product(exposed, detuned)
    pair = concatenate(exposed, detuned)
    with pytest.raises(ValueError, match="carrier"):
        feedback_reduce(pair, output="g.0", input="d.0")


def test_concrete_jax_operands_stay_in_jax() -> None:
    """Concrete JAX payloads are composed with JAX, not coerced to NumPy."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    a = _lowering()
    system = replace(_system("g", np.eye(2), [0.2 * a, 0.1 * a @ a], None), scattering=jnp.eye(2, dtype=complex))
    other = _system("o", np.eye(2), [0.3 * a, 0.2 * a @ a], None)
    stacked = concatenate(system, other)
    assert isinstance(stacked.S, jax.Array)
    for result in (series_product(system, other), feedback_reduce(stacked, output="g.0", input="o.0")):
        assert isinstance(result.S, jax.Array)
        assert all(isinstance(operator.values, jax.Array) for operator in result.L)


def test_thermal_input_survives_series_and_feedback_at_its_input_column() -> None:
    """SLH composition retains a surviving external bath and rejects an overwritten one."""
    first = _system("first", np.eye(2), [_lowering(), _lowering()], None)
    first = replace(first, channels=(replace(first.channels[0], input_occupation=0.3), first.channels[1]))
    second = _system("second", np.eye(2), [_lowering(), _lowering()], None)
    cascaded = series_product(first, second)
    assert [channel.input_occupation for channel in cascaded.channels] == [0.3, None]
    combined = concatenate(first, second)
    assert [channel.input_occupation for channel in combined.channels] == [0.3, None, None, None]
    reduced = feedback_reduce(first, output="first.0", input="first.1")
    assert reduced.channels[0].input_occupation == 0.3
    with pytest.raises(ValueError, match="independent thermal"):
        series_product(second, first)
