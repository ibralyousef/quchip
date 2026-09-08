```{raw} html
<p class="docs-eyebrow">quchip documentation</p>
```

# Start with a chip.

```{container} docs-intro
Model superconducting quantum devices in Python. Declare devices, couplings,
controls, and losses, then calculate spectra, simulate pulses, or fit parameters.
```

## Install

Requires Python 3.11 or newer.

```bash
python -m pip install quchip
```

See {doc}`installation <get-started/installation>` for optional backends and integrations.

## Your first calculation

Couple a transmon to a resonator and read their dressed frequencies.
Frequencies and couplings are in GHz; this model uses the rotating-wave approximation.

```python
from quchip import RWA, Capacitive, Chip, DuffingTransmon, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
chip = Chip(
    [q, r], [Capacitive(q, r, g=0.05, label="qr")], approximation=RWA()
)

print(f"Qubit: {chip.freq(q):.6f} GHz")
print(f"Resonator: {chip.freq(r):.6f} GHz")
```

```text
Qubit: 4.998533 GHz
Resonator: 7.001041 GHz
```

The coupling shifts each frequency from its bare value. Continue with
{doc}`your first chip <guides/defining-and-inspecting-a-chip>` to inspect the
Hamiltonian and change a parameter.

## What would you like to calculate?

```{raw} html
<div class="docs-paths">
  <a href="guides/statics-and-parameter-studies.html"><strong>Spectra and interactions</strong><span>Sweep parameters and follow dressed states.</span><b aria-hidden="true">→</b></a>
  <a href="guides/dynamics-pulses-and-readout.html"><strong>Pulses and readout</strong><span>Schedule drives, compare leakage, and read observables.</span><b aria-hidden="true">→</b></a>
  <a href="guides/steady-state-and-vna.html"><strong>Microwave response and loss</strong><span>Measure reflection through a fridge and explore Purcell filtering.</span><b aria-hidden="true">→</b></a>
  <a href="guides/differentiability.html"><strong>Gradients and fitting</strong><span>Differentiate observables and infer model parameters.</span><b aria-hidden="true">→</b></a>
</div>
```

Browse all {doc}`guides <guides/index>` or look up the
{doc}`API and physics conventions <reference/index>`.

```{toctree}
:maxdepth: 1
:hidden:

get-started/index
guides/index
reference/index
contribute/index
```
