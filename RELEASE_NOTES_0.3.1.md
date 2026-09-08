# quchip 0.3.1

Changes since [v0.3.0](https://github.com/quchip/quchip/compare/v0.3.0...v0.3.1).

## Measurements and fridge noise

- `VNA.measure()` captures the driven steady-state response, internal mode amplitudes and photon numbers, and output noise spectra. Receiver bandwidth, integration time, calibration and repeated IQ sampling reuse those results without another physical solve.
- Fridge noise propagates through declared attenuators, isolators, circulators, filters and amplifiers. Measurements report output noise density and contributions by source, including cross-output IQ covariance.
- `result.measure()` samples saved quantum states after either Schrödinger or master-equation evolution. Joint measurements preserve correlations and support assignment errors and calibrated conditional IQ distributions.
- `result.iq_readout()` uses captured downstream wiring; `IQReadout.from_wiring()` uses a supplied wired model. Both transform supplied conditional coherent fields and accumulate receiver noise without requiring a readout pulse. These detector models do not infer IQ signals from qubit populations or add measurement backaction to the simulated evolution.

## Documentation

- Added section navigation and aligned page titles across headings, sidebars and the README.
- Extended the fridge guide with steady-state versus sampled resonator responses and Rabi counts and IQ using the same wiring.
- Moved the Purcell calculation into Focused studies, retaining its existing URL.
- Shortened the cookbook to practical API choices and common pitfalls; example-authoring guidance now lives under Contribute.

## Public names and compatibility

Public names now distinguish bath occupation, jump rates and network ports:

| Previous name | Preferred name |
|---|---|
| `thermal_population` | `thermal_occupation` |
| `result.collapse_flux(...)` | `result.jump_rate(...)` |
| `component.side(...)`, `block.side(...)` | `component.port(...)`, `block.port(...)` |
| `network.exposure(...)`, `network.exposures` | `network.external_port(...)`, `network.external_ports` |
| `FieldExposure`, `FieldSide` | `NetworkPort`, `ComponentPort` |

The previous names remain compatibility aliases through 0.4 and are scheduled
for removal in 0.5. Old thermal constructor arguments, parameter bindings,
noise configurations and saved device dictionaries are accepted. Parameter
discovery and new serialized dictionaries use `thermal_occupation` only;
supplying both spellings in one update raises an error. The value is the bath's
mean occupation, not an initial qubit population. Jump rates include absorption
and dephasing channels and are not generally emitted photon fluxes.

The new measurement results are named `StateMeasurement` and `StateSamples`
for saved quantum states, and `VNAMeasurement`, `VNAMeasurementStatistics` and
`VNAMeasurementSamples` for VNA calculations. `result.measure(...)` and
`VNA.measure(...)` keep their existing call syntax. `fit_a_dress` is unchanged.
