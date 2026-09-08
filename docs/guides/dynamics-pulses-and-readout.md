```{include} ../../examples/00_hello_chip.md
:start-after: <!-- reader-content -->
```

## Measure a prepared state

A Rabi experiment can stop at state preparation. If the readout dynamics are
outside the question, sample the final state's Born probabilities and describe
the detector through its calibration. This works for kets from `sesolve` and
density matrices from `mesolve`.

Here a resonant square pulse rotates a two-level transmon through one Rabi
period. The model is closed, so quchip selects `sesolve`.

```python
import numpy as np
from quchip import ChargeDrive, Chip, DuffingTransmon, IQReadout, QuantumSequence, Square

q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
rabi_chip = Chip([q], frame="rotating")
xy = ChargeDrive(q, label="xy")
rabi_chip.wire(xy)
rabi = QuantumSequence(rabi_chip)
rabi.schedule(xy, envelope=Square(duration=40.0, amplitude=0.025), freq=5.0)
result = rabi.simulate(tlist=np.linspace(0, 40, 81), check_truncation=False)

measurement = result.measure(q)
counts = measurement.sample(256, seed=7).counts()
```

`measurement.probabilities` follows `measurement.outcomes`: `(0,)`, `(1,)`
for this qubit. The default basis is the isolated energy basis captured by the
solve. `result.measure(q1, q2)` retains joint probabilities and correlations;
it does not draw each qubit independently. Pass `basis="solver"` for solver
coordinates, or unitary columns in the captured energy basis for another
local measurement. Custom columns act in the stored integration frame;
phase-sensitive bases must be rotated explicitly when comparing lab-frame
and rotating-frame states. Batch measurements keep their sweep axes.

A calibrated assignment matrix gives the probability of recording each label
conditional on the physical outcome. Columns sum to one. In this example,
10% of ground preparations are recorded as excited, and 20% of excited
preparations are recorded as ground.

```python
assignment = np.array([[0.90, 0.20], [0.10, 0.80]])
recorded = measurement.recorded_probabilities(assignment)
counts = measurement.sample(256, assignment=assignment, seed=7).counts()
```

For an IQ record, supply the conditional centers and covariances from a detector
calibration. The following illustrative calibration uses arbitrary IQ units.
Its covariance already includes the apparatus noise. Each shot selects a Born
outcome, then draws from that outcome's IQ distribution.

```python
detector = IQReadout(means=[-0.8+0.15j, 0.8-0.15j], iq_covariance=np.eye(2)*0.06)
superposition = result.measure(q, t=10.0)
iq_shots = superposition.sample(1000, readout=detector, seed=7)
iq_mean = detector.mean(superposition.probabilities)
iq_covariance = detector.covariance(superposition.probabilities)
```

The left panel samples independently terminated experiments at saved Rabi
times. The right panel shows the conditional IQ mixture after a π/2 pulse.
Colors identify the simulated physical outcomes; IQ points remain unclassified.

<details>
<summary>Plot the counts and IQ samples</summary>

```python
from pathlib import Path
import matplotlib.pyplot as plt

root = next(path for path in (Path.cwd(), *Path.cwd().parents)
            if (path / "docs/_static/quchip.mplstyle").exists())
plt.style.use(root / "docs/_static/quchip.mplstyle")
figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), layout="constrained")
times = np.asarray(result.times)
probability = np.asarray(result.population(q, 1))
axes[0].plot(times, probability, color="#16181C", label="Born probability")
axes[0].plot(times, 0.10 + 0.70*probability, color="#C92F33", label="Detector probability")
saved_times = times[::4]
frequencies = [result.measure(q, t=t).sample(256, assignment=assignment, seed=20+i).counts()[(1,)]/256
               for i, t in enumerate(saved_times)]
axes[0].scatter(saved_times, frequencies, s=18, color="#246FA8", label="256 shots", zorder=3)
axes[0].set(xlabel="Pulse duration (ns)", ylabel="Excited fraction", ylim=(-0.04, 1.04), xlim=(0, 40))
axes[0].legend(fontsize=8, loc="upper right")
for outcome, color in enumerate(("#246FA8", "#C92F33")):
    points = iq_shots.iq[iq_shots.physical_indices == outcome]
    axes[1].scatter(points.real, points.imag, s=5, alpha=0.65, color=color,
                    edgecolors="none", label=f"Outcome {outcome}")
axes[1].set(xlabel="I (a.u.)", ylabel="Q (a.u.)", xlim=(-1.7, 1.7), ylim=(-1.1, 1.1))
axes[1].set_aspect("equal", adjustable="box")
axes[1].legend(fontsize=8, loc="upper center", ncol=2, markerscale=2)
figure.savefig(root / "docs/images/terminal_rabi.svg")
plt.close(figure)
```

</details>

```{figure} ../images/terminal_rabi.svg
:alt: Rabi probability, detector assignment bias and finite counts beside two calibrated IQ clouds.

Assignment errors change the recorded Rabi contrast. Finite shots fluctuate
around that detector probability. [PDF](../images/terminal_rabi.pdf)
```

A query with `t=` needs the state at that exact saved time. Final measurement
works with `states="final"`; earlier states require `states="all"`.
Measurements do not change the saved state or add another evolution step.
They do not represent continuous monitoring or mid-sequence feedback.
JAX probabilities and continuous detector moments remain differentiable;
use `key=` for compiled sampling, whose discrete labels have no gradient.

## Use fridge wiring

When conditional signals are known at the Markov boundary, the fridge can
supply their downstream gain and noise. Reuse `readout_chip` and its exposed
`readout` plane from the [fridge guide](steady-state-and-vna.md):

```{code-block} python
from quchip import IQReceiver

receiver = IQReceiver(integration_time=100_000)  # ns
wired_detector = IQReadout.from_wiring(
    readout_chip, readout, means=[-0.01, 0.01], frequency=6.5, receiver=receiver,
)
shots = result.measure(q).sample(1000, readout=wired_detector, seed=7)
```

These example means are conditional coherent fields in $1/\sqrt{\mathrm{ns}}$
at the selected channel **before** its output reference sections. The caller
supplies their relation to the measured outcomes. For mixed output paths,
`means={plane: centers, ...}` supplies several boundary fields; all unspecified
boundary fields are vacuum. Output delay assumes a stationary signal over the
acquisition window.

`IQReadout.from_wiring()` resolves the current wiring without quantum
evolution. If the preparation result already carries the fridge wiring,
`result.iq_readout(readout, ...)` uses that captured wiring, even if the chip
is edited later. Output amplifiers, filters and loss sections use the same noise
propagation and receiver integration as VNA. The returned detector's
`contributions` maps source names to integrated 2×2 IQ covariances, including
receiver vacuum. Changing the integration time reuses the wiring and runs no
quantum solver. Frequency-dependent sections require a resolved
`noise_frequencies` grid, with the same checks as VNA measurement.

This is a conditional coherent-field model. It omits occupied boundary inputs
and device-generated field correlations. Loads and losses inside the Markov
boundary already affect the preparation; their outgoing fluctuations need an
explicit field calculation or a detector calibration that includes them.
Use `IQReadout(...)` directly for such a calibration, without adding its
apparatus noise again. Use the explicit readout-pulse experiment above when
resonator dynamics, depletion or measurement backaction are part of the question.
