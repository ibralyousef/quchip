# An NV centre in diamond

A negatively charged nitrogen-vacancy (NV) centre is a spin-1 defect in
diamond. Its microwave resonances near 2.87 GHz move with magnetic field, so a
single centre works as a magnetometer at room temperature. Calculate those
resonances for the electron spin and its ¹⁴N nucleus, then simulate the two
measurements that read them: a pulsed ODMR scan and a Ramsey sequence.
Frequencies are in GHz, fields in mT and times in ns.

With the field $B$ along the NV axis, the ground-state spin Hamiltonian is

```{math}
H/h = D S_z^2 + \gamma_e B S_z + P I_z^2 - \gamma_n B I_z + A_\parallel S_z I_z,
```

where $S$ is the electron spin and $I$ the nitrogen spin, both spin 1.

| Parameter | Value |
|---|---:|
| Zero-field splitting $D$ | 2.870 GHz |
| Electron gyromagnetic ratio $\gamma_e$ | 28.03 GHz/T |
| ¹⁴N quadrupole splitting $P$ | −4.945 MHz |
| ¹⁴N gyromagnetic ratio $\gamma_n$ | 3.077 MHz/T |
| Axial hyperfine coupling $A_\parallel$ | −2.162 MHz |

The Hamiltonian follows [Doherty et al. (2013)](https://doi.org/10.1016/j.physrep.2013.02.001);
$P$ and $A_\parallel$ are from [Smeltzer et al. (2009)](https://doi.org/10.1103/PhysRevA.80.050302).
The transverse hyperfine term $A_\perp(S_xI_x+S_yI_y)$, with
$A_\perp=-2.70$ MHz ([Felton et al. 2009](https://doi.org/10.1103/PhysRevB.79.075203)),
connects levels about $D$ apart. At 10 mT it moves the resonances by at most
8 kHz, far below the linewidths resolved here, so it is omitted; the last
section checks this. A green laser prepares $m_s=0$ and reads the spin through its
fluorescence. That optical cycle is not modelled: each calculation starts in
$m_s=0$ and reports the $m_s=0$ population.

## Declare the electron and nitrogen spins

quchip's built-in models cover circuit QED; other systems enter as custom
devices. Both spins here have an axial splitting and a Zeeman
term, so one device class describes them. `CustomSpace` supplies the spin-1
matrices by name, and each `parameter` becomes a path in `chip.parameters`.
A spin 1 has exactly three levels: `validate` rejects any other `levels`, and
`truncation_boundary` returns `None` because there is no numerical cutoff.
The pulse calculations use the `basis` and `projection_levels` settings to
keep two electron levels. `physics_notes` records the omitted transverse
hyperfine term in `chip.physics_notes()`.

```python
from typing import Literal

import numpy as np
from scipy.signal import find_peaks
from quchip import (
    RWA, Chip, CouplingModel, CustomSpace, DeviceDrive, DeviceModel, LocalOps,
    QuantumSequence, Scalar, SpectrumSweep, Square, Sweep, parameter, setting,
)

SPIN_1 = {  # basis m = +1, 0, -1
    "I": np.eye(3, dtype=complex),
    "Sx": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2),
    "Sy": np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2),
    "Sz": np.diag([1.0, 0.0, -1.0]).astype(complex),
}


class Spin1(DeviceModel):
    """Spin 1 with an axial splitting and a Zeeman shift along the same axis."""

    _default_levels = 3
    basis: Literal["native", "eigen"] | None = setting(default=None, kw_only=True)
    projection_levels: int | None = setting(default=None, kw_only=True)
    splitting: Scalar = parameter(unit="GHz")
    zeeman: Scalar = parameter(unit="GHz/mT")
    field: Scalar = parameter(unit="mT")

    def local_space(self):
        return CustomSpace(3, SPIN_1)

    def validate(self):
        if self.levels != 3:
            raise ValueError(f"Spin1 has three levels; got levels={self.levels}.")

    def truncation_boundary(self):
        return None

    def local_hamiltonian(self, op, p):
        return p.splitting * op["Sz"] @ op["Sz"] + p.zeeman * p.field * op["Sz"]

    def transverse_spin(self):
        return LocalOps(label=self.label, space=self.local_space(), device=self)["Sx"]


class Hyperfine(CouplingModel):
    """Axial hyperfine interaction A S_z I_z."""

    A: Scalar = parameter(unit="GHz")

    def interaction(self, electron, nucleus, p):
        return p.A * electron["Sz"] * nucleus["Sz"]

    def physics_notes(self):
        return [*super().physics_notes(), "Secular hyperfine: A_perp (S_x I_x + S_y I_y) omitted"]
```

Place the centre in a 10 mT field:

```python
field = 10.0
nv = Spin1(splitting=2.870, zeeman=0.02803, field=field, label="nv")
nitrogen = Spin1(splitting=-0.004945, zeeman=-3.077e-6, field=field, label="n14")
chip = Chip([nv, nitrogen], [Hyperfine(nv, nitrogen, A=-0.002162, label="hf")])
```

`zeeman` is the shift per mT for each unit of $m$: $+\gamma_e$ for the
electron and $-\gamma_n$ for the nitrogen. Level labels follow each spin's
energy order. Between zero field and the level crossing near 102 mT, electron
level 0 is $m_s=0$, level 1 is $m_s=-1$ and level 2 is $m_s=+1$; at exactly
zero field, levels 1 and 2 are degenerate.

## Find the resonances

`transition_frequency()` diagonalizes the coupled spins and returns the energy
difference between two assigned eigenstates. Condition each electron
transition on the three nitrogen levels:

```python
resonances = {}
for name, level in (("0 -> -1", 1), ("0 -> +1", 2)):
    lines = [chip.transition_frequency(nv, 0, level, when={nitrogen: m}) for m in range(3)]
    resonances[name] = np.sort(lines)
    print(f"m_s = {name}: {', '.join(f'{f:.6f}' for f in resonances[name])} GHz")
```

Output:

```text
m_s = 0 -> -1: 2.587538, 2.589700, 2.591862 GHz
m_s = 0 -> +1: 3.148138, 3.150300, 3.152462 GHz
```

Each branch sits at $D\mp\gamma_eB$ and splits into three lines, one for each
nitrogen projection $m_I$, separated by $|A_\parallel|$. The nitrogen
quadrupole and Zeeman terms do not depend on $m_s$, so they cancel in these
transitions.

The same field acts on both spins. `Sweep.zip` moves their `field` parameters
together:

```python
fields = np.linspace(0.0, 60.0, 121)
sweep = SpectrumSweep(
    chip, [Sweep.zip(Sweep(fields, name="nv.field"), Sweep(fields, name="n14.field"))],
).run(progress=False)
ground = [sweep.energy_by_bare_label(nv=0, n14=m) for m in range(3)]
branches = {
    level: [sweep.energy_by_bare_label(nv=level, n14=m) - ground[m] for m in range(3)]
    for level in (1, 2)
}
```

<details>
<summary>Plot the resonances</summary>

```python
import shutil
import matplotlib.pyplot as plt

plt.style.use("../_static/quchip.mplstyle")
plt.rcParams["text.usetex"] = bool(shutil.which("latex"))

fig, axis = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
axis.axvline(field, color="#9AA0A8", ls="--", lw=1.0)
for level, color, label in ((1, "#246FA8", r"$m_s=0\rightarrow-1$"),
                            (2, "#C92F33", r"$m_s=0\rightarrow+1$")):
    for m, line in enumerate(branches[level]):
        axis.plot(fields, line, color=color, lw=1.8, label=label if m == 0 else None)
axis.set(xlabel="Field along the NV axis (mT)", ylabel="Resonance frequency (GHz)",
         xlim=(fields[0], fields[-1]))
axis.legend(loc="upper left")

fig.savefig("nv_resonances.svg")
plt.close(fig)
```

</details>

```{figure} ../images/nv_resonances.svg
:alt: The two NV electron-spin resonances start at 2.87 GHz and separate linearly with magnetic field.

The branches leave $D=2.87$ GHz at $\pm28.03$ MHz/mT. Each branch is three
hyperfine lines 2.2 MHz apart, which overlap at this scale. The dashed line
marks 10 mT. [PDF](../images/nv_resonances.pdf)
```

## Resolve the hyperfine lines with a π pulse

A weak microwave π pulse flips only the spins whose line it matches. Keep the
two electron levels it drives: at 10 mT the $m_s=+1$ transition is
$2\gamma_eB=561$ MHz away. The chip rotates each spin at its own transition
and applies the RWA to the microwave field. The dynamiqs backend, installed
with the `dynamiqs` extra, solves the batches of pulse sequences below
together.

```python
class MicrowaveLine(DeviceDrive):
    """Microwave magnetic field along x: H = gamma_e B_x(t) S_x."""

    def hamiltonian(self, spin, signal):
        return signal.i * spin.transverse_spin()


electron = Spin1(splitting=2.870, zeeman=0.02803, field=field,
                 basis="eigen", projection_levels=2, label="nv")
driven = Chip(
    [electron, nitrogen], [Hyperfine(electron, nitrogen, A=-0.002162, label="hf")],
    frame="rotating", approximation=RWA(), backend="dynamiqs",
)
microwave = MicrowaveLine(electron, label="mw")
_ = driven.wire(microwave)
```

The microwave amplitude is $\gamma_eB_1$ in GHz. Since
$\langle0|S_x|{-1}\rangle=1/\sqrt2$, the Rabi frequency is
$f_R=\gamma_eB_1/\sqrt2$; with $f_R=0.5$ MHz a π pulse lasts 1 µs. Its
excitation profile is 0.80 MHz wide at half maximum, narrower than the
2.162 MHz hyperfine spacing, so the pulse resolves the three lines.

```python
center = resonances["0 -> -1"][1]
rabi = 0.0005

scan = QuantumSequence(driven)
pi_pulse = scan.schedule(
    microwave,
    envelope=Square(duration=1 / (2 * rabi), amplitude=np.sqrt(2) * rabi),
    freq=center,
)
```

The nitrogen splittings are far below $k_BT$ at room temperature, so treat the
¹⁴N spin as an equal mixture of its three projections; optical nuclear
polarization is not modelled. Start from each nitrogen level, average, and
scan the carrier across the $m_s=0\rightarrow-1$ branch:

```python
nitrogen_states = [driven.state({electron: 0, nitrogen: m}) for m in range(3)]
carriers = center + np.linspace(-0.006, 0.006, 121)
odmr = scan.simulate_batch(
    scan.vary("initial_state", nitrogen_states), pi_pulse.vary("freq", carriers),
    tlist=[0.0, 1 / (2 * rabi)], progress=False,
)
bright = np.asarray(odmr.population(electron, 0, reduce="last")).mean(axis=0)

dips, _ = find_peaks(-bright, height=-0.8)
print(f"Dips at {', '.join(f'{1e3 * (carriers[i] - center):.1f}' for i in dips)} MHz; "
      f"lowest m_s = 0 population {bright.min():.3f}")
```

Output:

```text
Dips at -2.2, 0.0, 2.2 MHz; lowest m_s = 0 population 0.653
```

<details>
<summary>Plot the scan</summary>

```python
fig, axis = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
for line in resonances["0 -> -1"]:
    axis.axvline(1e3 * (line - center), color="#9AA0A8", ls="--", lw=1.0)
axis.plot(1e3 * (carriers - center), bright, color="#246FA8")
axis.set(xlabel=f"Microwave detuning from {center:.4f} GHz (MHz)",
         ylabel=r"$m_s=0$ population after a $\pi$ pulse",
         xlim=(-6, 6), ylim=(0.6, 1.02))

fig.savefig("nv_pulsed_odmr.svg")
plt.close(fig)
```

</details>

```{figure} ../images/nv_pulsed_odmr.svg
:alt: Pulsed ODMR spectrum with three dips at the three nitrogen hyperfine lines.

Each dip transfers the third of the population with one nitrogen projection.
The dashed lines are the resonances from `transition_frequency()`.
[PDF](../images/nv_pulsed_odmr.pdf)
```

A dip reaches about $2/3$ because each nitrogen level holds a third of the
population. The side lobes come from the square pulse's spectrum. This scan
omits the ¹³C bath introduced next, which would broaden each dip.

## Record Ramsey fringes

A Ramsey sequence π/2 – τ – π/2 lets the spin precess freely between two short
pulses. Detune both pulses 3 MHz below the central line and use $f_R=10$ MHz,
so each 25 ns pulse covers all three lines.

In natural diamond, ¹³C nuclei (1.1 %) produce a field that stays static
during one sequence but changes between repetitions. Take its detuning to be
Gaussian with $T_2^*=2$ µs, which gives a standard deviation
$\sigma=\sqrt2/(2\pi T_2^*)\approx113$ kHz. Shifting both carriers by
$-\delta$ is equivalent to shifting every line by $+\delta$, so zipped carrier
axes on nine Gauss–Hermite nodes sample the distribution.

```python
t2_star = 2000.0
sigma = np.sqrt(2) / (2 * np.pi * t2_star)
nodes, weights = np.polynomial.hermite_e.hermegauss(9)
weights = weights / weights.sum()
carrier = center - 0.003

ramsey = QuantumSequence(driven)
half_pi = Square(duration=25.0, amplitude=np.sqrt(2) * 0.010)
first = ramsey.schedule(microwave, envelope=half_pi, freq=carrier)
wait = ramsey.delay(electron, 100.0)
second = ramsey.schedule(microwave, envelope=half_pi, freq=carrier)
```

Vary the nitrogen level, the static offset and the free-precession time:

```python
delays = np.arange(20.0, 3001.0, 20.0)
fringes = ramsey.simulate_batch(
    ramsey.vary("initial_state", nitrogen_states),
    ramsey.zip(
        first.vary("freq", carrier - sigma * nodes, name="first carrier"),
        second.vary("freq", carrier - sigma * nodes, name="second carrier"),
    ),
    wait.vary("duration", delays),
    tlist=[0.0, delays[-1] + 2 * half_pi.duration], progress=False,
)
populations = np.asarray(fringes.population(electron, 0, reduce="last"))
signal = np.einsum("nbd,b->d", populations, weights) / 3
print(f"Batch shape: {populations.shape}")
```

Output:

```text
Batch shape: (3, 9, 150)
```

The axes are the nitrogen level, the static offset and the delay; `signal`
averages over the first two. A shared two-point `tlist` lets dynamiqs
integrate all 4,050 sequences at once. Populations do not change after the
second pulse, so the last time reads every delay.

Compare the fringe's Fourier peaks with the carrier's detuning from each line:

```python
amplitude = np.abs(np.fft.rfft(signal - signal.mean(), n=8192))
frequency = 1e3 * np.fft.rfftfreq(8192, d=delays[1] - delays[0])
peaks, _ = find_peaks(amplitude, height=0.5 * amplitude.max())
expected = 1e3 * (resonances["0 -> -1"] - carrier)
print(f"Fourier peaks: {', '.join(f'{frequency[i]:.2f}' for i in peaks)} MHz")
print(f"Line detunings: {', '.join(f'{f:.2f}' for f in expected)} MHz")
```

Output:

```text
Fourier peaks: 0.85, 3.00, 5.18 MHz
Line detunings: 0.84, 3.00, 5.16 MHz
```

<details>
<summary>Plot the fringes and their spectrum</summary>

```python
without_bath = populations[:, len(nodes) // 2].mean(axis=0)

fig, (trace, spectrum) = plt.subplots(2, 1, figsize=(7.2, 6.2), layout="constrained")
trace.plot(delays / 1e3, without_bath, color="#9AA0A8", lw=1.2,
           label=r"Without the $^{13}$C bath")
trace.plot(delays / 1e3, signal, color="#16181C", lw=1.6, label=r"$T_2^*=2\ \mu$s")
trace.set(xlabel=r"Free precession $\tau$ ($\mu$s)", ylabel=r"$m_s=0$ population",
          xlim=(0, 3), ylim=(0, 1))
trace.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2)

for line in expected:
    spectrum.axvline(line, color="#9AA0A8", ls="--", lw=1.0)
spectrum.plot(frequency, amplitude / amplitude.max(), color="#C92F33")
spectrum.set(xlabel="Fringe frequency (MHz)", ylabel="Fourier amplitude",
             xlim=(0, 8), ylim=(0, 1.05))

fig.savefig("nv_ramsey.svg")
plt.close(fig)
```

</details>

```{figure} ../images/nv_ramsey.svg
:alt: Ramsey fringes of the NV electron beating at three frequencies and decaying with the carbon-13 bath, with their Fourier spectrum.

Top: the $m_s=0$ population after the second π/2 pulse, with and without the
static ¹³C field. Bottom: the Fourier amplitude of the averaged fringe; dashed
lines mark each hyperfine line's detuning from the carrier.
[PDF](../images/nv_ramsey.pdf)
```

The fringe beats at three frequencies, one per nitrogen projection: the
carrier's detuning from each hyperfine line, $3$ MHz and $3\pm2.162$ MHz. The
Fourier peaks agree with them to 0.02 MHz, well within the 0.33 MHz resolution
of a 3 µs record. The decay comes from the ¹³C bath. A field change $\delta B$
shifts all three frequencies by $\gamma_e\delta B$, 28 kHz per µT, which is
the signal a Ramsey magnetometer records.

## Check the approximations

Restore the transverse hyperfine term and compare the six resonances:

```python
class FullHyperfine(CouplingModel):
    """Hyperfine interaction A S_z I_z + A_perp (S_x I_x + S_y I_y)."""

    A: Scalar = parameter(unit="GHz")
    A_perp: Scalar = parameter(unit="GHz")

    def interaction(self, electron, nucleus, p):
        transverse = electron["Sx"] * nucleus["Sx"] + electron["Sy"] * nucleus["Sy"]
        return p.A * electron["Sz"] * nucleus["Sz"] + p.A_perp * transverse


full_hyperfine = Chip([nv, nitrogen],
                      [FullHyperfine(nv, nitrogen, A=-0.002162, A_perp=-0.00270, label="hf")])
shift = max(
    abs(full_hyperfine.transition_frequency(nv, 0, level, when={nitrogen: m})
        - chip.transition_frequency(nv, 0, level, when={nitrogen: m}))
    for level in (1, 2) for m in range(3)
)
print(f"Largest resonance shift from A_perp: {1e6 * shift:.1f} kHz")
```

Output:

```text
Largest resonance shift from A_perp: 7.9 kHz
```

The shift is second order in $A_\perp/D$, far below the 0.80 MHz π-pulse
profile and the 0.33 MHz Fourier resolution. Near the level crossing at
102 mT, where $m_s=0$ and $m_s=-1$ meet, the transverse term mixes them and
must be kept.

The projected electron misses one physical effect. During a pulse, the
off-resonant $m_s=+1$ transition shifts $m_s=0$ by the AC Stark shift
$f_R^2/(4\cdot2\gamma_eB)\approx45$ kHz; over one 25 ns pulse that is a phase
of 7 mrad. Repeat three delays of the fringe with the full spin-1 electron:

<details>
<summary>Compare with the full spin-1 model</summary>

```python
full = Chip(
    [nv, nitrogen], [Hyperfine(nv, nitrogen, A=-0.002162, label="hf")],
    frame="rotating", approximation=RWA(), backend="dynamiqs",
)
full_line = MicrowaveLine(nv, label="mw")
_ = full.wire(full_line)

models = {"two-level": (driven, electron, microwave), "spin-1": (full, nv, full_line)}
checks = {}
for name, (model, spin, line) in models.items():
    sequence = QuantumSequence(model)
    _ = sequence.schedule(line, envelope=half_pi, freq=carrier)
    gap = sequence.delay(spin, 100.0)
    _ = sequence.schedule(line, envelope=half_pi, freq=carrier)
    states = [model.state({spin: 0, nitrogen: m}) for m in range(3)]
    result = sequence.simulate_batch(
        sequence.vary("initial_state", states),
        gap.vary("duration", np.array([20.0, 500.0, 1000.0])),
        tlist=[0.0, 1050.0], progress=False,
    )
    checks[name] = np.asarray(result.population(spin, 0, reduce="last"))

change = np.abs(checks["spin-1"] - checks["two-level"]).max()
sampled = weights @ np.cos(2 * np.pi * sigma * np.outer(nodes, delays))
decay_error = np.abs(sampled - np.exp(-(delays / t2_star) ** 2)).max()
print(f"Largest change with m_s = +1 included: {change:.4f}")
print(f"Nine-node error in the Gaussian decay: {decay_error:.1e}")
```

Output:

```text
Largest change with m_s = +1 included: 0.0042
Nine-node error in the Gaussian decay: 1.5e-05
```

</details>

At the three sampled delays, including $m_s=+1$ changes the $m_s=0$
population by at most 0.0042, on the scale of that phase. The ODMR pulse
sees a Stark shift of 0.1 kHz. The projection therefore holds for the pulses
used here; stronger drives, or fields where the two transitions approach each
other, need the full spin. The nine Gauss–Hermite nodes reproduce the
Gaussian decay $e^{-(\tau/T_2^*)^2}$ to $1.5\times10^{-5}$.

This model leaves out the optical cycle, crystal strain and electron
relaxation ($T_1$ of a few ms at room temperature). A field component
perpendicular to the NV axis adds $\gamma_e(B_xS_x+B_yS_y)$ and mixes the $m_s$
states. To include it, add transverse-field parameters to the device and
those terms to `local_hamiltonian`; `SPIN_1` already provides `"Sy"`.
