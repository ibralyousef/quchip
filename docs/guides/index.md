# Guides

Choose a physical question. Each guide starts with a runnable calculation,
then develops the model and interprets its observables.

New to quchip? Start with {doc}`your first chip <defining-and-inspecting-a-chip>`.

## Spectra and dynamics

- {doc}`Spectra and parameter sweeps <statics-and-parameter-studies>`:
  compare dressed observables and track states through avoided crossings.
- {doc}`Pulses and readout <dynamics-pulses-and-readout>`:
  compare pulse bandwidth, leakage, and conditional resonator response.

## Microwave networks

- {doc}`Resonator reflection <steady-state-and-vna>`:
  follow the measured signal through the fridge wiring.
- {doc}`Purcell filtering and T1 <slh-networks>`:
  compare circuits and identify where an excitation is lost.

## Reduction and inference

- {doc}`Chip transformations <chip-transformations>`:
  reduce a model and replay its controls.
- {doc}`Gradients and fitting <differentiability>`:
  differentiate observables and fit shared model parameters.

## From the talk

The {doc}`SQA 2026 companion <from-sqa-2026>` collects five short calculations
from the presentation. The guides above stand on their own.

```{toctree}
:hidden:
:maxdepth: 1

Spectra and sweeps <statics-and-parameter-studies>
Pulses and readout <dynamics-pulses-and-readout>
Resonator reflection <steady-state-and-vna>
Purcell filtering and T1 <slh-networks>
Chip transformations <chip-transformations>
Gradients and fitting <differentiability>
SQA 2026 companion <from-sqa-2026>
```
