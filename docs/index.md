```{image} _static/quchip-wordmark-light.png
:class: only-light
:width: 380px
:align: center
:alt: quchip
```

```{image} _static/quchip-wordmark-dark.png
:class: only-dark
:width: 380px
:align: center
:alt: quchip
```

# Documentation

`quchip` is an open-source Python toolkit for modelling superconducting quantum chips.

Declare devices, couplings, controls, and losses once. Use the same chip for
dressed spectra, pulse experiments, model reduction, and JAX gradients.

## Install

quchip requires Python 3.11 or newer.

```bash
pip install quchip
```

Optional extras: `quchip[dynamiqs]` for the JAX-native backend, `quchip[viz]` for graph visualization, `quchip[scqubits]` for scqubits interoperability.

The {doc}`backend guide <guides/choosing-a-backend>` shows how to select QuTiP
or dynamiqs, choose an integration method, and set tolerances, step controls,
batching, and gradients.

## Learn quchip

1. {doc}`Declare and inspect a chip <guides/defining-and-inspecting-a-chip>`
2. {doc}`Sweep spectra and compare measurements <guides/statics-and-parameter-studies>`
3. {doc}`Schedule pulses and read observables <guides/dynamics-pulses-and-readout>`
4. {doc}`Measure resonator reflection through a fridge <guides/steady-state-and-vna>`
5. {doc}`Compare Purcell filtering and T1 <guides/slh-networks>`
6. {doc}`Reduce a chip and replay its controls <guides/chip-transformations>`
7. {doc}`Differentiate observables and fit parameters <guides/differentiability>`

The {doc}`cookbook` defines the conventions used by executable quchip examples.

```{figure} images/hello_qubit_drive_leakage.png
:width: 760px
:alt: Short and long Gaussian pulses with multilevel qubit populations
```

```{figure} images/hello_dispersive_readout_iq.png
:width: 560px
:alt: Conditional resonator IQ paths with emphasized final points
```

`quchip` uses GHz for ordinary frequencies, ns for time, and mK for temperature. The implemented conventions and approximations are recorded in the {doc}`physics reference <physics>`.

## Start from the SQA 2026 talk

The {doc}`post-talk page <guides/from-sqa-2026>` collects five runnable entry points.

The accompanying paper is [quchip: A Differentiable Toolkit for Modeling Quantum Devices](https://arxiv.org/abs/2607.17081) (arXiv:2607.17081); citation metadata is in the repository's [CITATION.cff](https://github.com/quchip/quchip/blob/main/CITATION.cff).

```{toctree}
:maxdepth: 1
:hidden:

guides/defining-and-inspecting-a-chip
guides/statics-and-parameter-studies
guides/dynamics-pulses-and-readout
guides/steady-state-and-vna
guides/slh-networks
guides/chip-transformations
guides/differentiability
guides/choosing-a-backend
guides/migrating-to-0.3
guides/from-sqa-2026
cookbook
extensions
physics
api
contributing
conduct
```

## Project

- [GitHub](https://github.com/quchip/quchip)
- [PyPI](https://pypi.org/project/quchip/)
- License: Apache-2.0
