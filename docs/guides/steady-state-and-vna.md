# Steady state and microwave ports

Use a steady-state solve when the resolved Lindblad generator is constant and
you want the state after transients have died away. Add `Port` objects when a
collapse channel is also a microwave input or output.

## Solve an undriven steady state

```python
from quchip import Chip, Resonator

r = Resonator(freq=6.0, levels=10, T1=80.0, label="r")
chip = Chip([r])

result = chip.steadystate(e_ops={r: r.number_operator()})

rho_ss = result.state
n_ss = result.expect(r)
print(result.residual, result.trace_error, result.positivity_error)
```

`result.state` is native to the selected backend. `reduced_state(device)`
keeps the backend type. A closed system with more than one stationary state
raises instead of choosing one density matrix.

QuTiP reports nullity and condition number by default through total Hilbert
dimension 16. Above that it still reports the sparse Liouvillian residual and
returns `None` for those two dense diagnostics. Set
`options={"diagnostic_max_dimension": 24}` to raise that diagnostic limit, or
set it to zero to disable dense rank diagnostics. dynamiqs checks uniqueness in
its JAX solve; a non-unique solve inside `jax.jit` produces an invalid (`NaN`)
state instead of an arbitrary stationary state.

Sweep declared parameters with the same paths used elsewhere:

```python
from quchip import Sweep

states = chip.steadystate_batch(
    Sweep([0.0, 0.05, 0.1], name="r.thermal_population"),
    e_ops={r: r.number_operator()},
)

occupations = states.expect(r)
```

## Separate internal loss from measured ports

```python
from quchip import Chip, PortNetwork, Resonator

r = Resonator(
    freq=6.8,
    levels=12,
    internal_quality_factor=200_000,
    label="readout",
)

network = PortNetwork(label="feedline")
input_port = network.port(
    "in",
    target=r,
    external_quality_factor=15_000,
)
output_port = network.port(
    "out",
    target=r,
    external_quality_factor=18_000,
)
chip = Chip([r], port_network=network)
```

The resonator owns its unobserved loss. Each port owns one accessible external
channel. For this model,

```text
kappa_total = 2 pi f / Q_internal + 2 pi f / Q_in + 2 pi f / Q_out.
```

A port with an explicit `rate` uses `1/ns`. A collective port can target
several devices and take one dimensionless operator on that joint support.

For an explicit microwave network, define the field graph first and attach it
as one boundary object:

```python
from quchip import PortNetwork

measurement_network = PortNetwork(label="measurement_line")
coupler = measurement_network.port(
    "coupler",
    target=r,
    external_quality_factor=15_000,
)
phase = measurement_network.phase_shift("phase", phase=0.12)
cable = measurement_network.delay("cable", duration=3.2)
measurement_network.cascade(coupler, phase)
measurement_network.connect(phase.output, cable.input_terminal("1"))
vna_plane = measurement_network.expose(
    "vna_plane",
    input=coupler,
    output=cable.output_terminal("2"),
)

measurement_chip = Chip([r], port_network=measurement_network)
resolved = measurement_chip.resolve()
print(resolved.slh.S)
print(resolved.slh.L)
```

For reflection readout through a circulator and isolator:

```python
network = PortNetwork(label="fridge")
port = network.port("chip_port", target=r, rate=0.04)
circulator = network.circulator("circ")
isolator = network.isolator("iso")
network.link(port, circulator.side(2))
network.link(circulator.side(3), isolator)
drive = network.expose("drive", at=circulator.side(1))
readout = network.expose("readout", at=isolator.side(2))
```

Use `link` for physical cabling; it joins consecutive sides in both directions
and traverses a two-sided component from side 1 to side 2. Use `cascade` for a
directional SLH series connection.

`PortNetwork` composes instantaneous scalar scattering with the port coupling
operators. Scattering mappings use `(output, input)` keys. A reciprocal
two-sided attenuator uses power transmission `eta`; each direction has
amplitude transmission `sqrt(eta)` and couples to one of two hidden vacuum
channels with amplitude `sqrt(1-eta)`, so the resolved scattering matrix
remains unitary. A reflected amplitude crossing the attenuator in both
directions acquires a net factor `eta`. `network.delay(...)` adds a two-sided
reference section, placed with `link` or `connect` like any other component.
The compiler peels adjacent sections from each exposure leg before forming the
Markovian `S`, `L`, and `H`.

Use `network.filter(...)` for a passive frequency-dependent reference section:

```python
def lowpass(frequency, *, cutoff, order):
    return 1.0 / (1.0 + 1j * (frequency / cutoff) ** order)


filter_network = PortNetwork(label="filtered_line")
filter_port = filter_network.port("chip_port", target=r, rate=0.04)
ir_filter = filter_network.filter("ir_filter", transfer=lowpass, cutoff=7.2, order=2)
filter_network.link(filter_port, ir_filter)
filtered_readout = filter_network.expose("readout", at=ir_filter.side(2))
```

The keywords become sweepable, differentiable paths such as
`network.component.ir_filter.cutoff`. Continuous-wave calculations apply
`H(f)` exactly; transient inputs use `H(f_carrier)`, while transient output
amplitude and photon flux use `H(f_c)` and `|H(f_c)|^2` at the channel's
rotating-frame carrier. A network containing a filter cannot be serialized
with `to_dict()` because its transfer is a Python callable.

## One-tone response

```python
import numpy as np
from quchip import VNA

vna = VNA(chip, planes=[input_port, output_port])
result = vna.sweep(np.linspace(6.75, 6.85, 501))

matrix = result.matrix
s11 = result.s11
s21 = result.s21
s_out_in = result.s(output_port, input_port)
```

`sweep()` computes every small-signal derivative between the selected planes
around the current fixed-tone operating point:

```text
S_ji(f) = d <b_out,j> / d beta_in,i  at beta_probe -> 0.
```

`result.matrix` has shape `(*sweep_axes, n_planes, n_planes)` and is indexed
`[..., output, input]` in `result.planes` order.

`VNA.sweep()` also accepts ordinary chip-parameter `Sweep` axes. Their names
are parameter paths listed by `chip.parameters`, and the result axes follow
the order passed before `frequency`. Each grid point rebinds the chip through
the same immutable parameter machinery as `chip.with_params(...)` and
`steadystate_batch`.

```python
from quchip import Sweep

frequency_map = vna.sweep(
    np.linspace(6.75, 6.85, 201),
    Sweep(np.linspace(6.7, 6.9, 41), name="readout.freq"),
)
```

For a passive-linear model without fixed pumps or stationary solver options,
quchip lowers the declared Hamiltonian, loss channels, and `PortNetwork` to a
mode-space transfer problem. Eight harmonic resonators therefore require an
eight-dimensional linear solve, not a density matrix on their product Hilbert
space. The same call falls back to the stationary Liouvillian when the model
contains Kerr terms, finite-level saturation, pumps, time dependence, or
operator forms that cannot be classified as passive and linear. At each
frequency, the passive-linear route uses one multi-right-hand-side mode-space
solve. The stationary route solves one pumped operating point, then uses one
shifted-Liouvillian factorization for every input plane.
`result.diagnostics` records `linear_response` or `stationary_resolvent` as the
solver.

The direct background remains the resolved network `S`, while `L` supplies the
mode coupling and damping. Reference sections transform each exposure leg
between the chip boundary and its declared reference plane, with a continuous-
wave factor `exp(+i 2π f τ)` per leg. `SParameterResult` contains the scattering
matrix, selected plane labels, and diagnostics; use `chip.steadystate()` when
the stationary density matrix is itself the requested result.

## Finite fields and transient outputs

Use an explicit coherent field simulation for finite-power spectroscopy,
ring-up, ring-down, reflection, transmission, or emitted wave packets:

```python
from quchip import QuantumSequence, Square

sequence = QuantumSequence(measurement_chip)
sequence.schedule(
    vna_plane.input,
    envelope=Square(duration=200.0, amplitude=0.02),
    freq=6.8,
)
transient = sequence.simulate(
    tlist=np.linspace(0.0, 300.0, 601),
    e_ops={vna_plane: vna_plane.output},
)

field = transient.output(vna_plane)
b_out = field.amplitude
quadrature_i = field.quadrature(phase=0.0)
photon_flux = field.photon_flux
```

The scheduled amplitude is the incoming field `beta` in
`sqrt(photons/ns)`, so `abs(beta)**2` is incident photon flux in photons/ns.
The field follows `b_out = S b_in + L`. `field.quadrature(phase=theta)` uses
`Re[exp(-i theta) <b_out>]`, so changing the analysis phase does not require
another solve. `field.photon_flux` is the normally ordered
`<b_out^dagger b_out>`. Values are reported at the exposure reference plane;
`field.raw_amplitude` and `field.raw_photon_flux` retain the Markov-boundary
traces before the outbound reference sections shift them.

## Two-tone response

A pump is a fixed port tone and owns its sweep axes. Call
`pump.vary("freq", values)` or
`pump.vary("amplitude", values, name="pump_power")`; `name=` becomes the result
axis name, and a frequency axis defaults to `<plane>.freq`.

```python
pump = vna.pump(qubit_port, freq=5.0, amplitude=0.02)

trace = vna.sweep(
    6.8,
    pump.vary("freq", np.linspace(4.8, 5.2, 401)),
)

map_2d = vna.sweep(
    np.linspace(6.75, 6.85, 201),
    pump.vary("freq", np.linspace(4.8, 5.2, 161)),
)
```

`vna.zip(...)` can also mix `pump.vary(...)` with a chip-parameter `Sweep`,
stepping through both element by element.

Use `vna.zip(...)` to pair pump frequency and amplitude point by point. A
dispersive model such as `CrossKerr` stays static in separate probe and pump
frames. If the chosen coupling, frame, and approximation leave dynamic terms,
the call raises and points to `QuantumSequence`. Periodic and Floquet steady
states are outside this API.

For a probe entering through a Purcell filter, the probe frame follows passive
`Capacitive` or `TunableCapacitive` exchange paths to the unported readout mode.
Diagonal `CrossKerr` edges do not tie the pump and probe frames together.

## Output spectrum and correlations

```python
spectrum = vna.output_spectrum(
    output_port,
    frequencies=np.linspace(-0.1, 0.1, 501),
)

g1 = vna.g1(output_port, delays=np.linspace(0.0, 200.0, 401))
g2 = vna.g2(output_port, delays=np.linspace(0.0, 200.0, 401))
cross_g2 = vna.g2(
    output_port,
    delays=np.linspace(0.0, 200.0, 401),
    input=input_port,
)
```

The spectrum contains the normally ordered fluctuation spectrum. Its
`coherent_flux` field records the carrier separately because the carrier is a
delta peak rather than a sampled spectral density. `g1` and `g2` use the full
output field, including fixed coherent input at that port. These three
backend-neutral routines currently form a dense Liouvillian and therefore cap
the total Hilbert dimension at 16. Backend-native correlation tools remain an
option above that limit. Pass `input=` to correlate distinct ports; the result
records both `input_port` and `output_port`.

## Pick a backend

QuTiP exposes several stationary algorithms and dense or sparse linear
solvers. dynamiqs uses quchip's direct JAX solve for stationary work and keeps
small-signal response and transient output traces differentiable. Both use the
same resolved SLH operators for damping, coherent input, mean output, spectra,
and correlations. See the
{doc}`backend guide <choosing-a-backend>` for the concrete solver options and
links to both projects.
