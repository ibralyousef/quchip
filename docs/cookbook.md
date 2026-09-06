# Cookbook

This page defines how quchip examples should read and the defaults users can
expect them to follow. Each example starts with the smallest runnable answer,
shows its output, then adds one physical or numerical idea at a time.

## The example contract

A quchip example should:

1. state the physical question;
2. run through public APIs as written;
3. show the result directly below the code that produced it;
4. make units, frames, approximations, and truncations visible when they affect
   the answer;
5. preserve the source declaration when sweeping, fitting, or transforming;
6. finish with a numerical or physical check tied to the reported observable.

Keep the first calculation small. A reader should get one useful result before
meeting batching, model reduction, custom extensions, or differentiation.
Group sections around a physical experiment. Combine wiring and measurement
when they explain the same signal; split independent studies into linked guides.
Keep cells focused: declare, schedule or sweep, solve, then inspect. Split
plotting from calculations and collapse all plotting code, styles, and long
receipts; keep the figures visible.
Check the rendered page: an executed plot or successful build alone does not
prove the reader can see the figure. Use the existing red, blue, and charcoal
palette, SVG figures with PDF downloads, and labelled magnitude/phase y axes
on one frequency plot for S parameters.

## Declare the physics first

Construct devices, couplings, and the chip before adding controls or analysis.
Keep guide cells flat: declare each chip directly, without `make_chip()` or
other model factories. Reserve functions for necessary reusable calculations,
such as a loss or objective.
Import supported classes from `quchip`; use submodule imports only for a public
optional backend or extension surface.

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
chip = Chip([q, r], [Capacitive(q, r, g=0.05, label="qr")])
```

Use object references while constructing and scheduling. Give every component
a short stable label; labels become parameter paths such as `q.freq` and
`qr.g`.

quchip uses ordinary GHz for frequencies, couplings and energy divided by h,
ns for time, mK for temperature, and `1/ns` for Lindblad rates. Put units in prose, output labels,
and plot axes rather than encoding them in variable names alone.

## Make resolution choices inspectable

Defaults are fine when they do not affect the lesson. Set the frame and
approximation explicitly before interpreting rotating terms, dropped bands, or
solver dynamics.

Use the inspection surfaces in this order:

```python
description = chip.describe()
authored_latex = chip.unresolved_hamiltonian().latex()
canonical_latex = chip.hamiltonian().latex()

resolved = chip.resolve(frame="auto")
frame = resolved.resolved_frame
frame_plan = frame.plan
approximation = resolved.approximation
dropped_terms = resolved.dropped_terms_summary()
```

- `describe()` shows the declared components, units, frame, approximation, and
  Hilbert-space size.
- `unresolved_hamiltonian()` preserves the authored device and coupling
  expressions.
- `hamiltonian()` shows the canonical expression after basis, frame,
  truncation, and approximation policies.
- `resolve()` carries the basis transforms, frame frequencies, approximation,
  dropped terms, and collapse terms used by the backend.

Call `.matrix()` only when the numerical array is itself needed. A named
operator such as `H_0` in resolved LaTeX is an opaque numerical leaf, not
missing physics; read the authored expression and the resolution record beside
it.

## Distinguish bare declarations from dressed observables

Device constructor values describe local models. Ask the `Chip` for quantities
of the coupled system:

```python
bare_frequency = q.freq
dressed_frequency = chip.freq(q)
conditional_resonator_frequency = chip.freq(r, when={q: 1})
```

Device labels are fixed at construction. To use a different label, construct a
replacement device. Local queries such as `q.hamiltonian()` remain independent
of chip membership.

Use dressed transitions for resonant carriers unless the example intentionally
studies a bare detuning. `chip.freq()`, `transition_frequency()`,
`dressed_anharmonicity()`, `dispersive_shift()`, and `static_zz()` dress the
chip when needed; do not call `dress()` first merely as setup.

Use `kerr_matrix()` when the question involves several self-Kerr and
cross-Kerr coefficients:

```python
kerr = chip.kerr_matrix()
kerr.labels
kerr[q, r]
```

The axes follow `chip.devices`. Diagonal entries are
$E(2_i)-2E(1_i)+E(0)$; off-diagonal entries are
$E(1_i,1_j)-E(1_i)-E(1_j)+E(0)$. Both are dressed quantities in ordinary
GHz. They need not equal similarly named component parameters because every
device and interaction in the chip contributes to the dressed energies. A
device with fewer than three resolved levels has `NaN` on its diagonal. Under
the `KerrCavity` convention $H=\omega n-Kn(n-1)$, the diagonal is $-2K$.

Prepare states with `chip.state()` by default. Use `chip.bare_state()` only
when the question concerns a bare product state, and say why.

## Choose the smallest public surface

| Question | Start with |
|---|---|
| One dressed observable | `chip.freq()` or another `Chip` analysis method |
| A few changed parameters | `chip.with_params()` |
| A spectrum over a grid | `SpectrumSweep` and `Sweep` |
| A stationary state or occupation | `chip.steadystate()` or `steadystate_batch()` |
| Small-signal scattering | `VNA.sweep()` |
| Stationary finite-power response | `VNA.finite_power()` |
| A microwave network | `PortNetwork` |
| One pulse experiment | `QuantumSequence.simulate()` |
| Related pulse experiments | `QuantumSequence.simulate_batch()` |
| A smaller effective model | `eliminate()` or `active_patch()` |
| Independent connected components | `chip.partition()` |
| Bare parameters from dressed targets | `fit_a_dress()` |
| A scalar derivative | `jax.grad()` |
| A residual or trace derivative | `jax.jacrev()` or `jax.jacfwd()` |

Static examples should not create a `QuantumSequence`. Dynamic examples should
not replace a declared device or coupling with a hand-written effective matrix
unless the comparison itself is the subject.

For field measurements, declare coupling ports and expose the external ports.
Use `port.input` for a scheduled coherent input and `port.output` for a
requested field observable; a coupling port's terminals are wiring endpoints.
Input amplitude is in `sqrt(photons/ns)`. Output mean amplitude, photon flux,
and collapse jump rate are different quantities. A VNA sweep measures a
small-signal derivative; `finite_power()` solves a stationary finite probe and
`QuantumSequence` resolves transients. All stationary routes require a constant
resolved generator. See [steady states and VNA](guides/steady-state-and-vna.md)
for resonator reflection and staged fridge wiring, and
[Purcell filtering and the T1 budget](guides/slh-networks.md) for channel-resolved losses.

## Rebind instead of mutating

`with_params()` returns a new chip or sequence. Use it for sweeps, local design
changes, and differentiable parameter maps:

```python
shifted = chip.with_params({"q.freq": 5.1, "qr.g": 0.045})
```

`chip.parameters` lists bindable fields, including inactive optional values
such as `q.T1 = None`. Activate noise with concrete values before compiling a
sweep or gradient, then vary numerical values within that fixed structure:

```python
noisy = chip.with_params({"q.T1": 100.0, "q.T2": 150.0})
```

Joint device, envelope and signal-transform constraints apply to the complete update. For example,
change a pulse's ramp length and duration in the same `with_params()` call.
Sequence pulse handles and dotted paths refer to the same parameters; two batch
axes cannot vary the same parameter, even with different display names.

Keep the original object and show that its parameters did not move when this
matters to the example. Use `clone()` for a structural copy and
`to_dict()`/`from_dict()` for a declared-model round trip.

## Sweep calibrated controls

For `FluxTunableTransmon`, `freq` is the calibrated local frequency at
`flux_bias`. Together they anchor the SQUID dispersion. Rebinding only
`flux_bias` preserves that calibration and retunes the Hamiltonian, so sweep
the physical device coordinate directly:

```python
import numpy as np

from quchip import SpectrumSweep, Sweep

flux_values = np.linspace(0.0, 0.3, 101)
sweep = SpectrumSweep(
    chip,
    [Sweep(flux_values, name="coupler.flux_bias")],
).run(progress=False)
```

Set `freq` and `flux_bias` in the same `with_params()` call when defining a new
calibration anchor. Use `frequency_at()` to inspect another bias without
changing the chip and `flux_for_frequency()` for the inverse question on the
monotonic lobe. Convert laboratory voltage or pulse amplitude to reduced flux
before passing it to the device.

For independently built experiments, pass their requests to `solve_many`:

```python
from quchip import solve_many

results = solve_many([sequence_a.build_problem(duration=20.0),
                      sequence_b.build_problem(duration=40.0)])
```

Each request keeps its captured model, dimensions, grid, solver settings and
backend. Compatible requests share execution. The returned collection preserves
input order; inspect individual results when grids or native array backends
differ. `chip.solve_many(...)` remains the convenience for requests built by
that chip. A native `SolveBatch` keeps the structural requirements of compiled
parameter sweeps.

## Fit bare parameters from dressed targets

`fit_a_dress(desired)` treats component numbers as dressed constraints and
returns the corresponding bare chip as `fit.chip`. Common devices target their
dressed frequency and anharmonicity. A `Capacitive` edge between two
non-computational modes targets the dressed exchange rate. Other `Capacitive`
edges and `CrossKerr` couplings target the full cross-Kerr
$E_{11}-E_{10}-E_{01}+E_{00}$. These values are targets, not starting bare
coupling strengths. Add pair observables with `constraints=` and use `vary=`
only when the component defaults are not the parameters you want to move:

```python
fit = fit_a_dress(
    desired,
    constraints={(q0, q1): {"exchange_rate": -0.0022}},
)
```

Start with `print(fit.summary())`. It reports each target and its source, each
bare parameter and starting-point choice, the final loss, and the scaled
Jacobian rank and condition number. Use `fit.final_targets`,
`fit.parameter_reports`, and `fit.solver_info` for programmatic inspection.
`fit.history` records the loss at distinct residual evaluations; plot
`numpy.minimum.accumulate(fit.history)` for a monotone best-so-far curve because
numerical-Jacobian probes also appear in the raw history.

The automatic plan raises when it has too few targets or converges with a
rank-deficient final Jacobian. An explicit `vary=` plan may be intentionally
ambiguous; in that case the fit returns with a warning and records the weak parameter
directions in `fit.solver_info`.

`fit.converged` and `fit.message` expose the optimizer's outcome and termination
reason. A stopped fit retains its candidate and loss for inspection.
If its Jacobian lacks rank, it also warns and records the deficient directions.

In 0.3, `constraints`, `vary` and `start` replace the deprecated fitting
keywords. Put desired dressed values on the input model, add numerical
constraints, and select any nondefault free parameters with `vary`.
For example, a previous half-pull target `chi_target` becomes:

```python
fit = fit_a_dress(
    desired,
    constraints={edge: {"cross_kerr": 2 * chi_target}},
    vary={q: ("freq", "anharmonicity"), r: ("freq",), edge: ("g",)},
)
```

Full cross-Kerr and static-ZZ targets keep their signed values. Constraints
extend component defaults; use `None` to remove a default when replacing it
with another observable, such as
`{edge: {"cross_kerr": None, "coupling_strength": bare_g}}`.

Fitting evaluates the full model by default. Use `evaluator="local"` to
choose one-hop neighborhoods explicitly; this omits more distant devices and
their indirect effects. Each target report records the choice.
`max_hilbert_dim` limits the dimension of each evaluated model and raises before
optimization when exceeded. Increasing the limit permits a larger calculation.
Partial local extraction of a `PortNetwork` model is unsupported.

## Build pulse schedules from physical controls

Wire a public drive to its target, schedule an envelope, and keep the returned
pulse handle when a later batch varies that pulse.

```python
from quchip import ChargeDrive, Gaussian, QuantumSequence

line = ChargeDrive(q, label="xy")
chip.wire(line)

sequence = QuantumSequence(chip)
pulse = sequence.schedule(
    line,
    envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=0.04),
    freq=chip.freq(q),
)
```

`sequence.charge(..., freq=None)` captures `chip.freq(target)` from the sequence's
chip when scheduling the pulse. Later device edits do not retune it. Use an explicit
carrier or rebind `pulse.<index>.freq` to change an existing pulse. The device's
`reference_freq` setting controls frame/readout interpretation separately:
`None` resolves the chip's dressed reference for each new calculation.

Use channel cursors, `delay()`, and `barrier()` for serial timing. Supply
`start_time` only for intentional overlap. Put global phase on
`schedule(..., phase=...)`; keep envelope parameters responsible for waveform
shape.

For related experiments, vary the pulse handle or `initial_state` and call
`simulate_batch()`. Zip duration and amplitude when calibration requires them
to move together. Avoid a manual Python loop of independent solves when the
batch API represents the same study.

## Choose the simulated interval

`sequence.simulate()` uses the scheduled duration. For an idle experiment,
provide `duration=...`; it specifies an interval starting at zero, in ns.
A duration can extend a pulse schedule but cannot cut it short. Empty schedules
require a duration or an explicit `tlist`. The same choices apply to
`build_problem()`, `build_batch()`, and `simulate_batch()`.

Automatic sampling uses resolved Hamiltonian and decay scales, pulse features,
and the frequencies needed to reconstruct requested observables and native
cutoff populations. Stationary calculations can return only the interval
endpoints; a narrow pulse adds local samples instead of refining the entire
idle interval. Fixed-duration sweeps share one grid covering every point.
These are sampling heuristics: verify sensitive results against a finer grid.
Solver tolerances remain separate.

Envelope models expose local `sampling_times()`. Built-ins resolve Gaussian
widths and ramp edges; custom shapes can override this method to include narrow
features. The default probes 65 equally spaced local times and cannot certify
an arbitrary waveform. Custom opaque `TimeCoefficient` implementations require
an explicit grid. Automatic sampling also needs concrete numerical values:
choose a grid with `sequence.build_problem().tlist` outside JAX tracing and
reuse it through `tlist=...` while differentiating. Choose that reference model
to cover the frequencies, rates and pulse widths in the optimization range.

An explicit `tlist` is passed to the numerical solver unchanged and defines the
actual interval. `initial_state` belongs to its first time. For example,
`tlist=[20.0, 21.0, 25.0]` starts evolution at 20 ns, without evolving the
initial state through the first 20 ns. Pulses and carriers keep their original
absolute times. To include earlier evolution, start the calculation earlier
and select the retained results afterward. `duration` and `tlist` are mutually
exclusive.

Solver steps are separate from these returned coordinates. Dynamiqs receives
pulse discontinuities on the absolute clock; QuTiP resolves envelope edges and
uses a pulse-width step cap unless overridden. A sparse explicit grid therefore
does not disable pulse-aware integration. It can still miss features in returned
traces and sampled truncation checks. Sweep points with different durations keep
their own grids; inspect each result's `times`. `batch.times` and stacked
histories require equal grids, including under JAX tracing. Otherwise, inspect
individual results or use a per-point reduction such as `reduce="last"`.
Reductions run on each point's own samples before stacking; a final value may
therefore belong to a different final time at each point.

## Query saved times

Ordinary observable accessors return native arrays. Select values afterward:

```python
population = result.expect("q")
saved = result.observable_at(result.times[1], population)
nearest = result.observable_at(12.5, population, method="nearest")
selected = result.observable_at([12.5, 15.0], population, method="interpolate")
```

Query times must lie inside the simulated interval. Exact lookup is the default;
nearest selection breaks ties toward the earlier sample. Linear interpolation
works on real or complex values, with time on the last axis. Leading value axes
are preserved and followed by the query shape. Interpolation only uses saved
samples; it cannot recover an excursion missed by the grid. Dynamiqs queries
preserve JAX arrays and gradients, including time derivatives within a linear
interpolation segment. Invalid traced queries raise during execution.

`result.state_at(t)` retrieves a state at an exact saved time, or uses explicit
`method="nearest"`. State queries take one time and never interpolate kets or
density matrices. A final-only result can return its retained state at the final
saved time; earlier unsaved states remain unavailable. Partitioned results also
support these queries; requesting a joint state materializes the tensor product.

## Let the declared model choose the equation

QuTiP is the default backend. With no explicit solver, quchip uses `sesolve`
for a closed model and `mesolve` when devices, drives, couplings, or baths
contribute collapse channels, or when the initial state is a density matrix.
Adding a resonator quality factor or a bath can therefore change the equation
without changing the pulse schedule. An explicit `solver="sesolve"` rejects
active dissipation or a density-matrix initial state. To study Hamiltonian-only
evolution of an open model, pass `dissipation=False`; network-generated
Hamiltonian terms remain, and the result records the exclusion.

For a time-independent QuTiP problem with no explicit solver method, quchip
uses diagonal propagation at all requested save times when the total Hilbert
dimension is at most 64 for `sesolve`. For `mesolve`, the Liouvillian dimension
(the square of the Hilbert dimension) must be at most 1024, so the Hilbert
dimension may be at most 32. `diag` does not use adaptive tolerances or step
controls. Supplying `atol`, `rtol`, `nsteps`, or `max_step` keeps adaptive
integration; combining these controls with an explicit `method="diag"` raises.
Driven problems, larger spaces, network-generated static terms, and an explicit
non-`diag` method also keep adaptive integration. For dynamiqs, configure a native
method through `options={"method": ...}`; its default is `Tsit5`. Tolerances belong
on that method object. Unsupported options raise, and `result.stats["options"]`
records the effective native settings for either backend.

See {doc}`Backend and solver options <guides/choosing-a-backend>` for available
methods, option syntax, batching, and gradient controls.

## Check the retained space

Simulation checks the maximum boundary population over its sampling times,
including with `states="final"` or `states="none"`. Fock ladders use the highest
Fock level; charge and phase grids use both edges. Energy projection adds a check
of the highest retained energy state. The check is independent of readout
reference frequencies and requested `e_ops`.

This is a warning heuristic. It can miss excursions between samples and does not
estimate truncation error. Increase the relevant cutoff and compare observables.
Use `truncation_threshold=...` to adjust the warning threshold or
`check_truncation=False` to disable automatic sampling and checking. A retained
full history can still supply a later `result.check_truncation()`; a final state
alone cannot establish the maximum over time.

Components declare their cutoff through `truncation_boundary()`, returning a
`TruncationBoundary(indices, description, convergence_hint)` in their authored
basis. Intrinsically finite models return `None`. Unknown custom cutoffs produce
an explicit unavailable-diagnostic warning. The former generic `top_levels`
argument is removed; boundary selection belongs to the component model. Under
JAX tracing, boundary values remain differentiable, but host warning thresholds
cannot be evaluated; check the concrete result afterward.

## Read results at the level of the question

Use:

- `population(device, level)` for occupation of an isolated local energy state;
- `expect()` for a declared observable trace;
- `overlap()` for a target state;
- `reduced_state()` for one subsystem;
- `check_truncation()` before trusting a small local basis.

`population()` and `overlap()` return NumPy arrays with QuTiP and JAX arrays
with Dynamiqs, in eager and compiled code. Use `np.asarray(...)` when you
want host values. These are the primary methods; the duplicate
`population_array()` and `overlap_array()` methods have been removed.

Default `X`, `Y`, and `Z` observables use the lowest two isolated energy
states. Ground has `Z=+1`, excited has `Z=-1`, and the equal positive
superposition has `X=+1`. Higher levels contribute zero, so these expectations
also reflect leakage. Model parameters can change those local energy states;
built requests and results keep the vectors captured for their calculation.

Read a resolved collapse-channel jump rate from the stored states with:

```python
t1_flux = result.collapse_flux("hidden.q.thermal_emission")  # <L†L>(t), 1/ns
```

Weight each channel's integrated jump count by its signed contribution to the
excitation balance: `+1` for a lowering jump, `+2` for two-photon loss, `-1`
for thermal absorption, and `0` for dephasing. With these weights,
`n(0) = n(T) + Σ_c w_c ∫<L_c†L_c> dt` closes only when the Hamiltonian
conserves excitation number and there is no coherent injection.

Pass `e_ops=chip.e_ops(...)` when expectation traces are the main output.
Stored states remain available by default; disable them only when the memory
tradeoff is intentional.

Do not narrate a calculation with a series of `print()` calls. In a notebook,
display a compact dict or result object. In a Markdown snippet, keep the print
if it helps copy-paste use and place the captured output immediately below it.

```python
summary = {
    "dressed_f01_ghz": float(chip.freq(q)),
    "chi_ghz": float(chip.dispersive_shift(q, r) / 2),
}
summary
```

Long examples may finish with a machine-readable receipt: a compact dict or
JSON record containing the parameters, outputs, checks, solver, and generated
figure paths from that run. A receipt records evidence; it does not replace the
reader-facing result shown earlier.

## Check the reported quantity

Choose checks from the physics and numerics of the example:

- add local levels and compare the reported transition, population, or shift;
- refine a time or sweep grid;
- verify an automatic derivative when its accuracy is the subject;
- read RWA dropped terms and compare their band amplitude with their frame
  frequency;
- read transformation validity before comparing reduced and full dynamics;
- inspect dressed-state assignment overlaps near hybridization.

Do not set a tolerance from the residual produced by the run being tested.
Derive it from solver accuracy, truncation convergence, an approximation
parameter such as `g / detuning`, or a stated physical estimate.

## Keep plots tied to observables

Make the smallest figure that answers the question. Label axes with units and
state what was traced out or conditioned on. Plot the scheduled envelope when
pulse shape explains the result. Use equal aspect ratio for an IQ plane and a
log scale only when the orders of magnitude matter.

Use `chip.plot_graph()` for connectivity. Its default labels are the declared
bare frequencies and coupling strengths, so drawing the graph stays cheap and
does not hide a diagonalization. Ask for `values="dressed"` when the dressed
transition frequencies and full-pull cross-Kerr values are the point of the
figure, or `values="both"` when comparing the model inputs with its dressed
observables. The topology itself does not change between these views.

Every committed figure must come from the code shown in the example. Record
its path in the final receipt so a rerun overwrites the documented artifact.

The paired Markdown guide must also contain the captured textual outputs from
its executed notebook. Run `python tools/sync_example_outputs.py` after
execution, and use `--check` in verification. Jupytext source parity alone is
not enough because plain Markdown does not preserve notebook outputs.

## Handle optional models and extensions explicitly

State the required extra before the first optional import, for example
`quchip[dynamiqs]` or `quchip[scqubits]`. After importing from another library,
inspect labels, truncations, basis projections, and interactions before using
the converted chip.

Write an extension only when shipped components cannot express the local
physics. Choose the public extension surface by ownership: device, coupling,
drive, envelope, signal transform, dissipation channel, local space, or model
mapping. Extension code returns symbolic quchip physics and must not branch on
a backend.

## Use specific names

Avoid `recipe` as a generic name for an example. Say what the object is:
example, procedure, pulse schedule, parameter study, fit, reduction, or bath
model.

`Bath.recipe` is the one established API use. It selects a built-in
collapse-channel model: `"thermal"`, `"collective_decay"`, or
`"correlated_dephasing"`. In prose and inspection output, call this the bath
model; retain `recipe` only when naming the constructor argument, attribute, or
serialized field.

## Examples that follow these conventions

- {doc}`Define and inspect a chip <guides/defining-and-inspecting-a-chip>`
- {doc}`Statics and parameter studies <guides/statics-and-parameter-studies>`
- {doc}`Dynamics, pulses, observables, and readout <guides/dynamics-pulses-and-readout>`
- {doc}`Chip transformations <guides/chip-transformations>`
- {doc}`Differentiability <guides/differentiability>`
