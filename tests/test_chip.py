"""Chip assembly and frame-spec tests checked against closed-form eigenvalue formulas."""

from __future__ import annotations

from quchip.approximations import RWA

import warnings

import numpy as np
import pytest

from quchip.backend.protocol import Backend
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive, Coupling
from quchip.control.drive import ChargeDrive
from quchip.declarative.expr import UnboundParameterError
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


class TestChipHamiltonian:
    """Verify system Hamiltonian eigenvalues against analytical formulas."""

    def test_hamiltonian_resolves_rwa_while_unresolved_preserves_authored_terms(self) -> None:
        """Resolved inspection applies chip RWA while unresolved inspection preserves the authored interaction."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.2)])

        unresolved = chip.unresolved_hamiltonian().matrix()
        resolved = chip.hamiltonian().matrix()

        assert unresolved.shape == resolved.shape == (9, 9)
        assert not np.allclose(unresolved, resolved)
        first_result = chip.resolve()
        assert chip.resolve() is first_result
        np.testing.assert_allclose(resolved, first_result.hamiltonian().matrix(), atol=1e-12)

        q.freq = 5.1
        assert chip.resolve() is not first_result

    def test_symbolic_chip_hamiltonian_exposes_real_terms_and_parameter_paths(self) -> None:
        """Chip inspection preserves authored device terms and structurally retained RWA exchange terms."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.02)])

        hamiltonian = chip.unresolved_hamiltonian()

        assert hamiltonian.shape == (12, 12)
        assert hamiltonian.parameter_paths() == (
            "q.freq",
            "q.anharmonicity",
            "r.freq",
            "cap_0.g",
        )
        latex = hamiltonian.latex()
        assert r"\omega_{q}\,\hat n_{q}" in latex
        assert r"\alpha_{q}" in latex
        assert r"(\hat a_{q} + \hat a^\dagger_{q})" in latex
        assert r"(\hat a_{r} + \hat a^\dagger_{r})" in latex

    def test_unbound_chip_only_requires_values_at_materialization(self) -> None:
        """A fully symbolic Chip remains inspectable and names every missing value on numerical use."""
        q = DuffingTransmon(levels=3, label="q")
        r = Resonator(levels=4, label="r")
        hamiltonian = Chip([q, r], [Capacitive(q, r)]).unresolved_hamiltonian()

        assert r"\omega_{q}" in hamiltonian.latex()
        with pytest.raises(
            UnboundParameterError,
            match=r"q\.freq, q\.anharmonicity, r\.freq, cap_0\.g",
        ):
            hamiltonian.matrix()

    def test_two_device_hamiltonian_no_coupling(self, backend: Backend) -> None:
        """Uncoupled two-device eigenvalues equal the tensor sums of each device's own eigenvalues."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r])
        H = chip.hamiltonian()
        evals = np.sort(np.linalg.eigvalsh(H.matrix(backend=backend)).real)

        q_evals = [0.0, 5.0, 9.75]
        r_evals = [0.0, 6.0, 12.0, 18.0]
        expected = np.sort([eq + er for eq in q_evals for er in r_evals])
        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_coupled_system_eigenvalue_perturbation(self, backend: Backend) -> None:
        """Coupling shifts eigenvalues by ~O(g) from the uncoupled tensor-sum values."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        coupling = Capacitive(q, r, g=0.02)
        chip = Chip(devices=[q, r], couplings=[coupling])
        H = chip.hamiltonian()
        evals = np.sort(np.linalg.eigvalsh(H.matrix(backend=backend)).real)

        q_evals = [0.0, 5.0, 9.75]
        r_evals = [0.0, 6.0, 12.0, 18.0]
        uncoupled = np.sort([eq + er for eq in q_evals for er in r_evals])

        # Ground state should be very close (second-order shift ~g²/Δ)
        assert abs(evals[0] - uncoupled[0]) < 0.01

        for i in range(len(evals)):
            assert abs(evals[i] - uncoupled[i]) < 0.1, (
                f"Eigenvalue {i}: coupled={evals[i]:.6f}, "
                f"uncoupled={uncoupled[i]:.6f}, "
                f"shift={abs(evals[i] - uncoupled[i]):.6f}"
            )

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

    def test_hamiltonian_resolves_the_selected_frame(self, backend: Backend) -> None:
        """Resolved Hamiltonian inspection applies the chip's selected frame."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])
        H_lab = chip.hamiltonian()
        chip.set_frame("rotating")
        H_rot = chip.hamiltonian()
        assert not np.allclose(H_lab.matrix(), H_rot.matrix())

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

    def test_invalid_frame_raises(self) -> None:
        """Invalid frame string raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])
        with pytest.raises(ValueError, match="frame string must be"):
            chip.set_frame("invalid")

    def test_float_and_dict_frame_specs(self) -> None:
        """set_frame accepts float and per-device dict specs."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=3, label="r")
        chip = Chip(devices=[q, r])
        chip.set_frame(5.2)
        assert chip.frame == 5.2
        chip.set_frame({"q": 5.0, r: 7.5})
        assert chip.frame["q"] == 5.0
        assert chip.frame["r"] == 7.5


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

    def test_invalid_label_raises(self) -> None:
        """Unknown device label raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        with pytest.raises(ValueError, match="Device 'nonexistent' not found"):
            chip.state(nonexistent=1)

    def test_fock_index_out_of_range_raises(self) -> None:
        """Fock index >= levels raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        with pytest.raises(ValueError, match="exceeds device dimension"):
            chip.state(q=3)

    def test_negative_fock_index_raises(self) -> None:
        """Negative Fock index raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        with pytest.raises(ValueError, match="must be >= 0"):
            chip.state(q=-1)

    def test_state_rejects_duplicate_mapping_and_keywords(self) -> None:
        """Providing the same label via mapping and kwargs is ambiguous."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        with pytest.raises(ValueError, match="Duplicate device specification"):
            chip.state({q: 1}, q=1)


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

    def test_string_shorthand_requires_set_state_order(self) -> None:
        """String shorthand raises ValueError unless set_state_order has configured a device order."""
        chip = self._chip()
        with pytest.raises(ValueError, match="set_state_order"):
            chip.bare_state("e0")

    def test_string_shorthand_wrong_length_rejected(self) -> None:
        """String shorthand must supply exactly one symbol per ordered device, or bare_state raises ValueError."""
        chip = self._chip()
        chip.set_state_order("q", "r")
        with pytest.raises(ValueError, match="has .* chars but .* devices"):
            chip.bare_state("e")
        with pytest.raises(ValueError, match="has .* chars but .* devices"):
            chip.bare_state("e11")

    def test_string_shorthand_rejects_unknown_symbols(self) -> None:
        """Unrecognized level symbols in string shorthand raise ValueError."""
        chip = self._chip()
        chip.set_state_order("q", "r")
        with pytest.raises(ValueError, match="Unknown level symbol"):
            chip.bare_state("xz")

    def test_set_state_order_requires_every_device(self) -> None:
        """set_state_order raises ValueError unless every chip device is included in the order."""
        chip = self._chip()
        with pytest.raises(ValueError, match="every device"):
            chip.set_state_order("q")

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

    def test_empty_rejected(self) -> None:
        """superposition() with no components raises ValueError."""
        chip = self._chip()
        with pytest.raises(ValueError, match="at least one"):
            chip.superposition()


class TestDeviceOperators:
    """Pauli operators follow the current local truncation."""

    def test_paulis_follow_levels_change(self) -> None:
        """All three Pauli operators re-compute after levels change."""
        q = Resonator(freq=6.0, levels=3, label="r0")
        _ = q.sigma_x
        _ = q.sigma_y
        _ = q.sigma_z
        q.levels = 5
        assert q.sigma_x.shape == (5, 5)
        assert q.sigma_y.shape == (5, 5)
        assert q.sigma_z.shape == (5, 5)


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


class TestConstructorLabelUniqueness:
    """Chip.__init__ rejects duplicate labels within each component kind."""

    def test_duplicate_bath_labels_raise(self) -> None:
        """Two baths sharing a label raise at construction."""
        from quchip.chip.baths import Bath

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        with pytest.raises(ValueError, match="[Dd]uplicate bath"):
            Chip(
                [q],
                baths=[
                    Bath("thermal", temperature=20.0, rate=1e-3, label="b"),
                    Bath("thermal", temperature=25.0, rate=1e-3, label="b"),
                ],
            )

    def test_duplicate_drive_labels_via_direct_control_equipment_raise(self) -> None:
        """Two drives sharing a label, wired via a directly-built ControlEquipment, raise at construction."""
        from quchip.control.equipment import ControlEquipment

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        d1 = ChargeDrive(target=q, label="d")
        d2 = ChargeDrive(target=q, label="d")
        with pytest.raises(ValueError, match="[Dd]uplicate drive"):
            Chip([q], control_equipment=ControlEquipment([d1, d2]))

    def test_duplicate_bath_labels_via_set_noise_raise(self) -> None:
        """Two baths sharing a label raise in set_noise before any mutation."""
        from quchip.chip.baths import Bath

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([q])
        with pytest.raises(ValueError, match="[Dd]uplicate bath"):
            chip.set_noise(
                baths=[
                    Bath("thermal", temperature=20.0, rate=1e-3, label="b"),
                    Bath("thermal", temperature=25.0, rate=1e-3, label="b"),
                ],
            )
        assert chip.baths == ()


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

    def test_from_array_accepts_string_device_label(self, backend: Backend) -> None:
        """from_array accepts a string device label equivalent to passing the device object."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([q], backend=backend)

        op = chip.from_array(np.eye(3, dtype=complex), device="q")
        assert (op - backend.identity(3)).norm() < 1e-12

    def test_from_array_validates_full_space_shape(self) -> None:
        """from_array with no device raises ValueError when the array shape mismatches the tensor-product space."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=2, label="r")
        chip = Chip([q, r])

        with pytest.raises(ValueError, match="full-space operator shape"):
            chip.from_array(np.eye(3, dtype=complex))
