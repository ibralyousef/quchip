# Migrating to quchip 0.3

quchip 0.3 changes model ownership, fitting, solver requests and retained
reductions. Recreate saved models from their Python declarations before
saving them in the new format.

## Model context and parameters

A device label is fixed after construction. Create a replacement device to
change its identity. Ask the chip for dressed quantities, for example
`chip.freq(q)`, rather than asking a device to discover its containing chip.
One device can participate in several models.

Use `chip.with_params({"q.freq": 5.1, "qr.g": 0.02})` for an independent
variant. Parameter paths are flat and shared by fitting, sweeps and binding.
Combined updates validate the final candidate, so joint changes to related
noise parameters do not depend on dictionary order. Built calculations and
results keep their captured model context.

Local state indices and populations refer to isolated energy levels. The
excited state has Pauli Z expectation −1; the plus state has X expectation
+1. Authored physical operators keep their declared coordinate meanings.

## Solver requests and results

Ordinary scheduled simulations can omit `tlist`. quchip constructs a grid
using the pulse support and resolved dynamics. An explicit `tlist` is passed
to the numerical solver as its requested time grid; the integrator can take
additional internal steps. Pulse boundaries and step controls remain separate
inputs. Use the backend's tolerances and step options for convergence studies.

`duration=` sets an automatic-grid interval and cannot be combined with
`tlist`. It must include the scheduled operations. Explicit `tlist` can select
a shorter interval; pulse times remain absolute.

Choose `states="all"`, `"final"` or `"none"` directly on the simulation request.
The default is `"all"`. `final_state` is available for both `"all"` and
`"final"`; `states` requires the complete retained history. Requested
observable traces are independent of state storage. Remove native
`store_states` options from old calls.

Use `state_at(t, method="exact")` or `method="nearest"` for saved states.
States are never interpolated. For observable values, use
`result.observable_at(t, result.expect("q"), method="interpolate")`; exact and
nearest lookup are also available. Queries outside the saved interval fail.

Use `dissipation=False` for a Hamiltonian-only calculation. This is an explicit
request choice and does not change the model's noise declaration. Unknown
solver names and options fail early. A failed numerical batch raises with the
point index and parameter values instead of returning a partially valid batch.

The truncation warning now checks sampled boundary occupation. Increase the
retained space to verify the observable; the warning is a heuristic.

## Fitting

Replace the retired `coupling_targets`, `observable_targets` and
`fit_parameters` interface with a desired model and `constraints`, `vary` and
`start`. Full-model evaluation is the default. Choose `evaluator="local"`
explicitly when its neighborhood approximation is appropriate; model size
never silently changes the fitting physics.

Desired cross-Kerr constraints use the full conditional transition difference,
`f_r(q=1) - f_r(q=0)`. If an old target used the sigma-Z half-pull convention,
its desired cross-Kerr is twice that signed value. Fit convergence and
identifiability remain distinct. CR-fit covariance is local and residual
scaled; unavailable covariance is not a zero error bar.

## Reductions and replay

Retained Hamiltonian matrices and transformed loss operators determine the
reduced model. Frequency and coupling summaries are diagnostics; they cannot
reconstruct every multilevel correction. `result.mapping` projects operators
and states into the retained space and lifts states back. An active patch
composes the maps of its reduction steps. Projection does not renormalize a
state: lost norm measures population outside the retained subspace.
Binding the returned chip edits that retained model. To recompute the
approximation for changed source parameters, bind the source first and rerun
`eliminate()`.

Controls and bath operators follow the retained coordinate change. General
time-dependent collapse channels remain unsupported. Field-aware elimination
supports a linear resonator's default port with one unprojected Fock-space
survivor; unsupported reductions raise.

## Extensions and saved models

Signal transforms use the same `parameter()` and `setting()` declarations as
other components. Implement `apply(signals)` and, when relevant,
`referenced_lines()`. Replace `_parameter_names` and routine field serialization
with those declarations. Add `serializable=True` to opt into saving a class,
and import the extension module before loading it. See the
[extension guide](../extensions.md) for the complete example.

Custom `ReductionMethod` strategies now supply `retained_hamiltonian(ctx)`
and `embedding(ctx)`: a retained Hamiltonian in GHz and an isometry from retained
to source coordinates. See [extensions](../extensions.md) for dimensions and
operator conventions.

Saved chips now carry `format_version: 1`. Unversioned or unsupported formats
are rejected; recreate older models from their declarations. The installed
package includes `py.typed` for external type checking. The scqubits extra
constrains QuTiP to the supported range; CI constraints target Linux. On macOS,
scqubits currently requires an older SciPy and a compatible JAX resolution;
install the extras together in a fresh environment.

## VNA port selection

Use `VNA(chip, ports=[p1, p2])` instead of `planes=...`. Omitting `ports`
selects all exposed instrument ports. The VNA and its scattering and mean-field
results expose the ordered labels as `.ports`, replacing `.planes`.
