# quchip 0.3.0

Changes since 0.2.1.

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

## Breaking changes

- Component labels are immutable. Create a replacement component to rename it.
- Local state indices and Pauli observables use isolated energy levels.
- Removed deprecated fitting arguments `coupling_targets`, `observable_targets`, and `fit_parameters`. Use `constraints`, `vary`, and `start`; full-model evaluation is the default.
- Set `states="all"`, `"final"`, or `"none"` on simulation requests instead of native `store_states` options. Explicit `tlist` specifies the solver grid; states are never interpolated.
- Updated signal-transform declarations and custom reduction hooks.
- Saved models require `format_version: 1`. Recreate older models from their Python declarations.

See the [migration guide](docs/guides/migrating-to-0.3.md) for replacements and reduction limits, and the [microwave guide](docs/guides/steady-state-and-vna.md) for supported network and response calculations.
