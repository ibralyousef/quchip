# Release notes

## Unreleased

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

```{include} ../RELEASE_NOTES_0.3.0.md
:relative-docs: docs/
:heading-offset: 1
```
