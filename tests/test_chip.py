"""Chip assembly and frame-spec tests checked against closed-form eigenvalue formulas."""

from __future__ import annotations

from quchip.approximations import RWA

import warnings

import numpy as np
import pytest

from quchip.backend.protocol import Backend
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive, Coupling
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


class TestChipHamiltonian:
    """Verify system Hamiltonian eigenvalues against analytical formulas."""


    def test_exact_zero_coupling_emits_no_rwa_vanish_warning(self) -> None:
        """A coupling whose interaction Hamiltonian is exactly zero builds silently, without the RWA-vanish warning."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r], couplings=[Capacitive(q, r, g=0.0)])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chip.hamiltonian()

    def test_nonzero_fully_rwa_rejected_coupling_warns(self) -> None:
        """A nonzero interaction whose every populated band the RWA predicate rejects still warns."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        # a⊗b + a†⊗b† is the two-mode-squeezing term: both populated bands
        # violate the default number-conserving predicate (Δa + Δb == 0).
        squeezing = Coupling(
            q,
            r,
            g=0.05,
            interaction=lambda a, b, bk: (
                bk.tensor(a.lowering_operator(), b.lowering_operator())
                + bk.tensor(bk.dag(a.lowering_operator()), bk.dag(b.lowering_operator()))
            ),
        )
        chip = Chip(devices=[q, r], couplings=[squeezing])

        with pytest.warns(UserWarning, match=r"vanishes entirely under RWA\(\)"):
            chip.resolve()


class TestFrameSpec:
    """Verify frame-spec APIs and frame-resolution behavior."""


    def test_resolve_accepts_a_frame_override_without_mutating_the_chip(self) -> None:
        """One resolution may use the lab frame without changing configured intent."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([q], frame="rotating")

        rotating = chip.resolve().hamiltonian().matrix()
        lab = chip.resolve(frame="lab").hamiltonian().matrix()

        assert chip.frame == "rotating"
        assert not np.allclose(rotating, lab)
        np.testing.assert_allclose(lab, chip.unresolved_hamiltonian().matrix(), atol=1e-12)
        np.testing.assert_allclose(
            q.resolve(frame="lab").hamiltonian().matrix(),
            q.unresolved_hamiltonian().matrix(),
            atol=1e-12,
        )

    def test_analysis_engine_result_is_the_static_spectral_contract(self) -> None:
        """Dressed analysis consumes an inspectable, cached lab-frame static result."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        chip = Chip(
            [q, r],
            [Capacitive(q, r, g=0.05)],
            frame="rotating",
            approximation=RWA(),
        )

        result = chip.analysis.engine_result()

        assert chip.analysis.engine_result() is result
        assert result.dynamic_terms == ()
        assert result.collapse_terms == ()
        assert "max_carrier_freq_ghz" not in result.metadata
        assert "max_step_ns" not in result.metadata
        expected = np.sort(np.linalg.eigvalsh(result.hamiltonian().matrix()).real)
        np.testing.assert_allclose(chip.dressed_spectrum(), expected, atol=1e-12)


class TestStateFactory:
    """Verify state factory builds correct tensor-product states."""

    def test_bare_state_accepts_device_keyed_mapping(self, backend: Backend) -> None:
        """bare_state({device: level}) matches the label-keyed form."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r])

        psi = chip.bare_state({q: 0, r: 1})
        dims = [3, 4]
        n_q = backend.embed(q.number_operator(), 0, dims)
        n_r = backend.embed(r.number_operator(), 1, dims)

        assert abs(backend.expect(n_q, psi)) < 1e-12
        assert abs(backend.expect(n_r, psi) - 1.0) < 1e-12

    def test_multi_excitation_state(self, backend: Backend) -> None:
        """bare_state(q=2, r=1) = |2>⊗|1>, matching both device excitation numbers."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r])

        psi = chip.bare_state(q=2, r=1)
        dims = [3, 4]
        n_q = backend.embed(q.number_operator(), 0, dims)
        n_r = backend.embed(r.number_operator(), 1, dims)

        assert abs(backend.expect(n_q, psi) - 2.0) < 1e-12
        assert abs(backend.expect(n_r, psi) - 1.0) < 1e-12


class TestStateOrderShorthand:
    """`chip.set_state_order(...)` + string shorthand for bare_state / state."""

    def _chip(self) -> Chip:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        return Chip(devices=[q, r])

    def test_string_shorthand_parses_letters_and_digits(self) -> None:
        """Level-order string shorthand parses letters and digits to the same state as an explicit index mapping."""
        chip = self._chip()
        q, r = chip["q"], chip["r"]
        chip.set_state_order(q, r)
        # "e2" → {q: 1 (e), r: 2}
        expected = chip.bare_state({q: 1, r: 2})
        actual = chip.bare_state("e2")
        import numpy as np

        diff = np.linalg.norm(np.asarray(chip.backend.to_array(expected - actual)))
        assert diff < 1e-12


    def test_string_shorthand_rejects_unknown_symbols(self) -> None:
        """Unrecognized level symbols in string shorthand raise ValueError."""
        chip = self._chip()
        chip.set_state_order("q", "r")
        with pytest.raises(ValueError, match="Unknown level symbol"):
            chip.bare_state("xz")


    def test_set_state_order_custom_levels(self) -> None:
        """Custom level-symbol mapping in set_state_order resolves string shorthand to its level indices."""
        chip = self._chip()
        chip.set_state_order("q", "r", levels={"a": 0, "b": 1, "c": 2})
        import numpy as np

        expected = chip.bare_state({"q": 1, "r": 2})
        actual = chip.bare_state("bc")
        diff = np.linalg.norm(np.asarray(chip.backend.to_array(expected - actual)))
        assert diff < 1e-12


class TestSuperposition:
    """`chip.superposition(...)` primitive."""

    def _chip(self) -> Chip:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        return Chip(devices=[q, r])

    def test_equal_two_component_matches_manual(self) -> None:
        """Default equal-weight superposition equals the manually summed and renormalized bare states."""
        chip = self._chip()
        a = chip.bare_state({"q": 0, "r": 0})
        b = chip.bare_state({"q": 1, "r": 0})
        manual = a + b
        manual = manual / chip.backend.norm(manual)
        psi = chip.superposition({"q": 0, "r": 0}, {"q": 1, "r": 0})
        import numpy as np

        diff = np.linalg.norm(np.asarray(chip.backend.to_array(psi - manual)))
        assert diff < 1e-12

    def test_weighted_superposition(self) -> None:
        """Weighted superposition with amplitude coefficients summing to unit probability normalizes to unit norm."""
        chip = self._chip()
        import numpy as np

        psi = chip.superposition(
            (np.sqrt(0.3), {"q": 0, "r": 0}),
            (np.sqrt(0.7), {"q": 1, "r": 0}),
        )
        # |c_00|^2 = 0.3, |c_10|^2 = 0.7 ⇒ unit norm
        assert abs(chip.backend.norm(psi) - 1.0) < 1e-12

    def test_accepts_string_shorthand(self) -> None:
        """superposition() built from string-shorthand components matches the equivalent dict-keyed components."""
        chip = self._chip()
        chip.set_state_order("q", "r")
        psi_str = chip.superposition("g0", "e0")
        psi_dict = chip.superposition({"q": 0, "r": 0}, {"q": 1, "r": 0})
        import numpy as np

        diff = np.linalg.norm(np.asarray(chip.backend.to_array(psi_str - psi_dict)))
        assert diff < 1e-12


class TestSubspaceAccessors:
    """Explicit Fock-basis accessors for qudits and multi-level devices.

    ``sigma_plus`` / ``sigma_minus`` project into ``{|0>, |1>}`` (the
    computational subspace); ``projector(i, j)`` / ``transition(i, j)``
    name the subspace explicitly.
    """

    def test_sigma_plus_cache_invalidates_on_levels_change(self) -> None:
        """sigma_plus cache is invalidated and rebuilt at the new dimension when levels changes."""
        q = Resonator(freq=6.0, levels=3, label="r0")
        first = q.sigma_plus
        q.levels = 5
        second = q.sigma_plus
        assert first.shape != second.shape
        assert second.shape == (5, 5)


class TestFromArray:
    """Verify Chip.from_array embeds local and full-space operators correctly."""

    def test_from_array_embeds_local_operator_for_device_object(self, backend: Backend) -> None:
        """from_array embeds a local operator for a device object at its tensor slot in the full chip space."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=2, label="r")
        chip = Chip([q, r], backend=backend)

        local = np.diag([0.0, 1.0, 2.0]).astype(complex)
        op = chip.from_array(local, device=q)
        expected = backend.embed(backend.from_array(local, dims=[[3], [3]]), 0, [3, 2])

        assert (op - expected).norm() < 1e-12


    def test_from_array_validates_full_space_shape(self) -> None:
        """from_array with no device raises ValueError when the array shape mismatches the tensor-product space."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=2, label="r")
        chip = Chip([q, r])

        with pytest.raises(ValueError, match="full-space operator shape"):
            chip.from_array(np.eye(3, dtype=complex))
