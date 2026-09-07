# quchip 0.3.0

Changes since [v0.2.1](https://github.com/quchip/quchip/releases/tag/v0.2.1).

## New features

- Added steady-state solves and batches with observables and numerical diagnostics.
- Added `PortNetwork` for coherent inputs, measured output fields, passive components, reference-plane delays, filters, and amplifiers. Includes reusable network blocks, SLH composition, instantaneous algebraic feedback, and network/S-parameter plots.
- Added VNA scattering matrices, phase-conjugate response, parameter and pump sweeps, finite-power spectroscopy, output spectra, and correlations.
- Added pulse-aware automatic time grids and `frame="auto"`.
- Added explicit state-storage choices, saved-state lookup, and observable interpolation.
- Added per-channel jump rates and cumulative jump counts from stored states.
- Added forward-mode JAX differentiation through local eigensystems.

## Improvements and fixes

- Accelerated eligible static QuTiP simulations with automatic diagonal propagation and passive-linear VNA calculations with mode-space solves.
- Fixed density-matrix solver routing and excluded cascades from automatic diagonal propagation.
- Made calculation snapshots independent of later model edits and parameter updates independent of binding order.
- Retained Hamiltonian corrections, controls, and loss channels through supported reductions; added composable state/operator maps.
- Corrected partitioning for interactions spanning multiple devices and made failed batches identify the failing point.
- Updated examples, backend guidance, and extension documentation. Added `py.typed` and removed the unused Optax dependency.

## Migrating from 0.2.1

- Labels are immutable; create a replacement component to rename one. Replace `device.dressed_freq` and chip-bound `device.drive_freq` with `chip.freq(device)`.
- Local state indices, populations, and Pauli operators use isolated energy levels; excited-state Z is −1.
- Replace `population_array()` / `overlap_array()` with `population()` / `overlap()` (NumPy on QuTiP, JAX on dynamiqs).
- Set `states="all"`, `"final"`, or `"none"` instead of native storage options. For the old nearest-time behavior, pass `method="nearest"` to `state_at()` / `dm_at()`; the default is now `"exact"`.
- `chip.parameters` includes unset optional fields as `None`. Skip them before numerical conversion; unchanged rebinding still works.
- Replace deprecated fitting arguments `coupling_targets`, `observable_targets`, and `fit_parameters` with `constraints`, `vary`, and `start`. Fitting defaults to `evaluator="full"`; choose `"local"` explicitly.
- Replace reduction metadata key `"folded_into"` with `"coupling"`. Keep the returned chip's effective terms; parameter summaries no longer reconstruct the reduction.
- Custom signal transforms use `parameter()` / `setting()` instead of `_parameter_names`; custom reductions implement `retained_hamiltonian(ctx)` and `embedding(ctx)`. See [extensions](docs/extensions.md).
- Saved models require `format_version: 1`; recreate older models from Python declarations.
