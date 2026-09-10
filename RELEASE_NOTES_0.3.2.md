# quchip 0.3.2

Changes since [v0.3.1](https://github.com/quchip/quchip/compare/v0.3.1...v0.3.2).

## Network noise API

- Passive network components now use `thermal_occupation`; replace previous
  `occupation` and `loss_occupation` arguments and saved parameter keys.
- Amplifiers require explicit `added_noise`. Temperature, noise-figure, and
  `noise_frequency` constructor arguments have been removed; convert these
  inputs to noise quanta before declaring components. See the
  [network-noise migration guidance](docs/cookbook.md#declare-network-noise).

## API documentation

- Added parameter options, defaults, units, `None` behavior, result shapes,
  and physics references across public constructors and methods.
- Reuse inherited backend contracts and suppress empty type-only parameter
  tables. A coverage check detects missing descriptions and stale names.
- Serve README images from the documentation site for consistent rendering.

## Development and releases

- PR fast and pre-merge test selections cover the suite without repeating the
  fast tests on Python 3.11. Merges require checks against current `main`.
- Remove post-merge test reruns, make benchmarks manual, and fail change
  classification when Git cannot read the compared revisions.
- Tagged releases verify package metadata and release notes, publish to PyPI,
  and create the corresponding GitHub Release after publication.
