# Backend and solver options

QuTiP is the default backend. Use dynamiqs for JAX gradients, compiled batches,
or accelerator execution. Both solve the declared model with its chosen frame,
approximation, controls, and loss channels.

## Choose a backend and equation

```python
chip = Chip(devices, couplings, backend="qutip")
result = sequence.simulate(duration=100.0, backend="qutip")
```

Install `quchip[dynamiqs]` and use `backend="dynamiqs"` on the chip or solve to
select that backend.

| Request | Automatic equation |
|---|---|
| Ket, no collapse channels | `sesolve`: Schrödinger equation |
| Density matrix or collapse channels | `mesolve`: Lindblad master equation |

Leave `solver` unset for this selection. An explicit `solver="sesolve"` rejects
collapse channels and density matrices. Use `dissipation=False` for an
intentional Hamiltonian-only calculation.

## Time grid, stepping, and saved states

Omit `tlist` for a pulse-aware grid; use `duration` to include an idle interval.
An explicit `tlist` is passed to the solver and defines the initial and final
times. Adaptive methods choose internal steps separately.

```python
result = sequence.simulate(
    tlist=times,
    e_ops=chip.e_ops(q="n"),
    states="final",
)
```

`states="all"` retains the history, `"final"` keeps only the final state, and
`"none"` keeps neither. Requested observables remain available in all three
cases. Post-solve state and collapse-flux queries need the relevant saved states.

## QuTiP integration

```python
result = sequence.simulate(
    tlist=times,
    options={"method": "vern9", "rtol": 1e-9, "atol": 1e-11},
)
```

| Calculation | Methods |
|---|---|
| General non-stiff dynamics | `adams` (default), `vern7`, `vern9`, `dop853`, `tsit5` |
| Stiff dynamics | `bdf`, `lsoda` |
| Small constant Hamiltonian or Liouvillian | `diag` |
| Large sparse constant closed system | `krylov` |

For a constant generator, quchip automatically selects `diag` up to Hilbert
dimension 64 for `sesolve` or 32 for `mesolve`, unless adaptive controls or another
method were supplied. Cascade-generated Hamiltonians retain adaptive integration.
`diag` cannot be combined with adaptive tolerances or step controls.

`rtol` and `atol` set error tolerances, `max_step` caps an internal step in ns,
and `nsteps` caps the step count. quchip supplies pulse-aware stepping limits;
an explicit `max_step` or `nsteps` overrides its corresponding limit.
`result.stats["options"]` records the effective settings.

## dynamiqs integration and gradients

Pass a native method object with its tolerances:

```python
import dynamiqs as dq

result = sequence.simulate(
    tlist=times,
    backend="dynamiqs",
    options={"method": dq.method.Dopri8(rtol=1e-9, atol=1e-11, max_steps=100_000)},
)
```

| Calculation | Methods |
|---|---|
| General differentiable dynamics | `Tsit5` (default), `Dopri5`, `Dopri8` |
| Stiff dynamics | `Kvaerno3`, `Kvaerno5` |
| Fixed-step reference | `Euler(dt=...)` |
| Rouchon master equation | `Rouchon1`, `Rouchon2`, `Rouchon3` |

Pulse boundaries are supplied as discontinuities. `max_steps` limits work; it
is not a step size. Trajectory/Monte Carlo methods are unsupported.

The default gradient mode supports reverse mode (`jax.grad`). For forward
mode (`jax.jacfwd` or JVPs), pass `"gradient": dq.gradient.Forward()` in
`options`. Keep graph structure, local dimensions, and term counts fixed
through a traced calculation. The first call compiles; matching later calls
can reuse that compilation.

## Batches

`simulate_batch()` groups compatible problems. QuTiP distributes larger batches
across worker processes; dynamiqs vectorizes homogeneous batches. Numerical
failure raises with the original point index and parameter values when available.
A failed batch does not return partial results.

## Stationary states

`chip.steadystate()` requires a constant resolved generator and a unique
stationary state. QuTiP accepts native stationary methods and linear solvers:

```python
stationary = chip.steadystate(options={"method": "direct", "solver": "spsolve"})
```

`direct` is the usual choice; `eigen`, `svd`, `power`, and `propagator` are also
available. dynamiqs uses a differentiable constrained direct solve. A non-unique
state raises outside tracing and becomes `NaN` inside `jax.jit`.

Inspect `residual`, `trace_error`, and `positivity_error`. Dense QuTiP nullity
and condition-number diagnostics are computed only through Hilbert dimension
16 by default; adjust `diagnostic_max_dimension` when needed.

Compare the observable at tighter tolerances and larger local spaces before
trusting a result. The [differentiability guide](differentiability.md) checks
gradients; the [readout guide](steady-state-and-vna.md) checks stationary and
transient field responses.
