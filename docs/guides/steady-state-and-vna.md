# Resonator readout and fridge wiring

Follow one three-resonator chip from its fridge wiring to a stationary VNA
response and sampled IQ. Frequencies are in GHz, times in ns, temperatures
in mK, and decay rates in `1/ns`. Run the cells in order.

## Define the resonators and their bus

Three harmonic resonators span 6–7 GHz and exchange photons with a 6.5 GHz
bus. Only the bus couples directly to the measurement line. Each internal Q
sets a loss rate; each design Qe sets the coupling to the lossy bus. The full
coupled response determines the shifted resonances and their linewidths.
These are illustrative parameters, not a fit to a measured chip.

```python
import numpy as np
from scipy.signal import butter
from quchip import Capacitive, Chip, IQReceiver, PortNetwork, Resonator, RWA, VNA, qnp

mode_frequencies = np.array([6.0, 6.45, 7.0])
internal_q = np.array([8000, 25000, 12000])
external_q = np.array([1800, 3200, 2200])
bus = Resonator(freq=6.5, levels=3, internal_quality_factor=100_000, label="bus")
resonators = [Resonator(freq=f, levels=3, internal_quality_factor=qi, label=f"r{i+1}")
              for i, (f, qi) in enumerate(zip(mode_frequencies, internal_q))]
bus_qe = 6.5
bus_external_rate = 2 * np.pi * bus.freq / bus_qe
bus_total_rate = bus_external_rate + 2 * np.pi * bus.freq / bus.internal_quality_factor
external_rates = 2 * np.pi * mode_frequencies / external_q
coupling_strengths = np.sqrt(external_rates * (
    (bus_total_rate / 2)**2 + (2 * np.pi * (mode_frequencies - bus.freq))**2
) / bus_external_rate) / (2 * np.pi)
couplings = [Capacitive(bus, r, g=g, label=f"bus_r{i+1}")
             for i, (r, g) in enumerate(zip(resonators, coupling_strengths))]
```

The bus has a 1 GHz external linewidth. The coupling formula uses each
resonator's isolated response through that bus to set a nominal Qe. The
simulation retains all three couplings, including their mutual effect through
the bus. RWA keeps the exchange terms. Harmonic stationary acquisition uses
mode-space equations, so the local `levels` do not truncate this calculation;
nonlinear devices retain the density-matrix solver.

## Wire the fridge and define its noise

Passive loss is specified in dB and its load temperature in mK. The HEMT uses
gain in dB and equivalent input noise temperature; the room amplifier uses
noise figure. The physical and equivalent temperatures have different meanings:

| Element | Declaration | Noise model |
|---|---|---|
| Attenuator or cable | `loss_db`, `temperature`, `noise_frequency` | Planck occupation at the physical load temperature |
| Isolator | Load `temperature`, plus a separate insertion loss | Reverse field terminates in the load; the load emits upstream |
| Circulator | Ideal routing, unless loss terminals are explicitly added | No dissipative noise from an ideal unitary component |
| Matched absorptive filter | Complex `transfer`, `loss_temperature` | Emission proportional to absorbed power, `(1 - abs(H)**2) * n` |
| HEMT | `gain_db`, `noise_temperature`, `noise_frequency` | Equivalent added noise `k_B * T_e / (h * f)` |
| Room amplifier | `gain_db`, `noise_figure_db`, `noise_frequency` | `T_e = 290 K * (10**(NF/10) - 1)` |

`gain` (linear power gain), `eta` (power transmission), and `added_noise`
(input-referred symmetrized quanta) remain available. Choose one convention
for each quantity. Noise temperature is **not** the HEMT's physical stage
temperature. The noise-frequency reference fixes the occupation over the
Markov band and remains a model parameter. For context, cryogenic amplifier
datasheets report [gain and equivalent noise temperature](https://lownoisefactory.com/wp-content/uploads/2026/02/lnf-lnc4_8sg.pdf);
[noise figure uses a 290 K reference](https://helpfiles.keysight.com/csg/pxivna/Applications/Noise_Figure.htm).

Both filters have half-power edges at 4 and 8 GHz. A fourth-order Butterworth
low-pass prototype gives an eighth-order bandpass. The input filter sees a
vacuum source; the thermal attenuators follow it, so their emitted noise reaches
the chip without passing through that filter. A colored thermal reservoir
feeding a quantum coupling requires an explicit dynamical filter/bath model.

```python
filter_b, filter_a = butter(4, [4.0, 8.0], btype="bandpass", analog=True)


def passband(frequency):
    return qnp.polyval(filter_b, 1j * frequency) / qnp.polyval(filter_a, 1j * frequency)


fridge = PortNetwork(label="thermal_fridge")
port = fridge.port("bus_coupler", target=bus, external_quality_factor=bus_qe)
input_filter = fridge.filter("input_4_8GHz", transfer=passband)
att_4k = fridge.attenuator("att_4K", loss_db=20, temperature=4000, noise_frequency=6.5)
att_cp = fridge.attenuator("att_CP", loss_db=20, temperature=100, noise_frequency=6.5)
att_mxc = fridge.attenuator("att_MXC", loss_db=20, temperature=20, noise_frequency=6.5)
circ = fridge.circulator("circ")
iso_1 = fridge.isolator("iso_1", temperature=20, noise_frequency=6.5)
iso_2 = fridge.isolator("iso_2", temperature=20, noise_frequency=6.5)
iso_loss = fridge.attenuator("iso_insertion", loss_db=1, temperature=20, noise_frequency=6.5)
output_filter = fridge.filter("output_4_8GHz", transfer=passband,
                              loss_temperature=20, noise_frequency=6.5)
coax = fridge.attenuator("output_coax", loss_db=2, temperature=4000, noise_frequency=6.5)
hemt = fridge.amplifier("HEMT_4K", gain_db=40, noise_temperature=2500, noise_frequency=6.5)
room_amp = fridge.amplifier("amp_RT", gain_db=20, noise_figure_db=3, noise_frequency=6.5)
fridge.link(input_filter, att_4k, att_cp, att_mxc, circ.side(1))
fridge.link(circ.side(2), port)
fridge.link(circ.side(3), iso_1, iso_loss, iso_2, output_filter, coax, hemt, room_amp)
drive = fridge.expose("drive", at=input_filter.side(1))
readout = fridge.expose("readout", at=room_amp.side(2))
readout_chip = Chip([bus, *resonators], couplings, port_network=fridge,
                    approximation=RWA(), frame="rotating")
```

The circulator routes the reflected signal through two ideal isolators with
1 dB combined insertion loss, a cold filter, 2 dB of cable loss at 4 K, and
the amplifiers. Internal Qi baths and the coherent source are vacuum. Thermal
input changes the harmonic stationary fluctuations; the mean remains linear.
Saturation, compression, and finite reverse isolation are not included.

<details>
<summary>Draw the same fridge wiring</summary>

```python
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

plt.style.use("../_static/quchip.mplstyle")
plt.rcParams["text.usetex"] = bool(shutil.which("latex"))
fig, axis = plt.subplots(figsize=(7.2, 6.6), layout="constrained")
axis.set(xlim=(0, 12), ylim=(-0.1, 10.8))
axis.axis("off")
for name, bottom, top in [("Room", 8.5, 10.8), ("4 K", 6, 8.5),
                          ("100 mK", 4.5, 6), ("20 mK", -0.1, 4.5)]:
    axis.axhspan(bottom, top, color="#F2F4F6" if bottom in (6, -0.1) else "#FAFBFC", zorder=0)
    axis.text(0.2, (bottom + top) / 2, name, fontsize=10, va="center")
axis.plot([3, 3, 6], [10, 3.0, 3.0], color="#246FA8", lw=1.6)
axis.plot([6, 9.5, 9.5], [3.0, 3.0, 10], color="#C92F33", lw=1.6)
axis.plot([6, 6], [2.7, 1.5], color="#16181C", lw=1.4)
axis.annotate("", (3, 8.5), (3, 9.0), arrowprops={"arrowstyle": "->", "color": "#246FA8"})
axis.annotate("", (9.5, 10.2), (9.5, 9.7), arrowprops={"arrowstyle": "->", "color": "#C92F33"})


def box(x, y, text, width=1.8):
    axis.add_patch(Rectangle((x-width/2, y-.3), width, .6, ec="#16181C", fc="white", zorder=3))
    axis.text(x, y, text, fontsize=8, ha="center", va="center", zorder=4)


box(3, 10, "Source + 4–8 GHz filter", width=3.1)
box(3, 7.3, "20 dB attenuation")
box(3, 5.2, "20 dB attenuation")
box(3, 3.7, "20 dB attenuation")
axis.add_patch(Circle((6, 3), .35, ec="#16181C", fc="white", zorder=3))
axis.text(6, 3, r"$1\to 2$" + "\n" + r"$2\to 3$", fontsize=7, ha="center", va="center", zorder=4)
axis.text(6, 3.55, "Circulator", fontsize=8, ha="center")
box(8.1, 3, "2 isolators\n1 dB total loss", width=2.3)
box(9.5, 4, "4–8 GHz filter")
box(9.5, 6.6, "2 dB cable loss")
box(9.5, 7.7, "HEMT: +40 dB\nNoise temp: 2.5 K", width=2.7)
box(9.5, 9.3, "Room amp: +20 dB\nNoise figure: 3 dB", width=2.7)
box(6, 1.5, "Bus: 6.5 GHz", width=2.5)
for x, frequency in zip((3.7, 6, 8.3), mode_frequencies):
    axis.plot([6, x, x], [1.2, .85, .4], color="#16181C", lw=.9)
    box(x, .3, f"{frequency:g} GHz", width=1.7)
fig.savefig("fridge_wiring.svg")
plt.close(fig)
```

</details>

```{figure} ../images/fridge_wiring.svg
:alt: Three resonators coupled to a bus, driven through staged fridge attenuation and a circulator, with two isolators, a 4–8 GHz output filter and two amplifiers.

One declared input and return path is used for both the VNA response and noisy
acquisition below. [PDF](../images/fridge_wiring.pdf)
```

## Calculate the stationary VNA response

The frequency grid covers the full band and resolves the narrow features.
The source amplitude is in `sqrt(photons/ns)`, before the 60 dB input loss.

```python
frequencies = np.unique(np.concatenate([
    np.linspace(5.97, 7.03, 401),
    *(f + np.linspace(-0.012, 0.012, 121) for f in mode_frequencies),
]))
vna = VNA(readout_chip, ports=[drive, readout])
response = vna.sweep(frequencies)
ideal = response.s(readout, drive)
```

The source and receiver are separate instrument ports, so the returned
reflection appears in S21. `ideal` is the complete stationary mean at the declared instrument planes,
including physical loss, filters, and gain.

## Acquire noise, then sample the measurement

Keep the same chip, fridge, frequencies, and instrument planes. `measure()`
captures the stationary mean and physical noise spectra. Its ratio agrees with
the harmonic S21 above. Choose integration time and sample count afterward.

```python
measurement = vna.measure(frequencies, amplitudes=20, input=drive, outputs=[readout])
short = measurement.sample(1, receiver=IQReceiver(integration_time=1_000_000), seed=19)
long = measurement.sample(1, receiver=IQReceiver(integration_time=10_000_000), seed=19)
np.testing.assert_allclose(measurement.ratio(readout), ideal, atol=1e-10)
```

The samples add jointly drawn IQ fluctuations from that same model. Increasing
integration from 1 ms to 10 ms reuses the stored spectra. The common seed
exposes the change in variance.

<details>
<summary>Plot the full band and all three resonances</summary>

```python
fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), sharex=True, sharey="row", layout="constrained")
ideal_phase = np.unwrap(np.angle(ideal))
for column, (samples, title) in enumerate(((short, "1 ms"), (long, "10 ms"))):
    observed = samples.ratio(readout)[0]
    axes[0, column].plot(frequencies, 20 * np.log10(np.abs(observed)), ".",
                         color="#C92F33", ms=2, alpha=0.6, label="Sampled IQ")
    axes[0, column].plot(frequencies, 20 * np.log10(np.abs(ideal)), color="#16181C",
                         lw=1.2, label="Ideal stationary mean")
    axes[1, column].plot(frequencies, (ideal_phase + np.angle(observed / ideal)) / np.pi,
                         ".", color="#C92F33", ms=2, alpha=0.6)
    axes[1, column].plot(frequencies, ideal_phase / np.pi, color="#16181C", lw=1.2)
    axes[0, column].set_title(title)
    axes[1, column].set_xlabel("Probe frequency (GHz)")
axes[0, 0].set_ylabel("Full-chain output / input (dB)")
axes[1, 0].set_ylabel(r"Phase / $\pi$")
axes[0, 1].legend(frameon=False, fontsize=8)
fig.savefig("fridge_measurement.svg")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=True, layout="constrained")
observed = long.ratio(readout)[0]
for index, (axis, frequency) in enumerate(zip(axes.flat, mode_frequencies)):
    selected = np.abs(frequencies - frequency) <= 0.012
    detuning = (frequencies[selected] - frequency) * 1000
    axis.plot(detuning, 20 * np.log10(np.abs(observed[selected])), ".", color="#C92F33", ms=3)
    axis.plot(detuning, 20 * np.log10(np.abs(ideal[selected])), color="#16181C", lw=1.2)
    axis.set_title(f"R{index+1}: {frequency:.3f} GHz", fontsize=10)
    axis.text(0.03, 0.04, f"Qi = {internal_q[index]:,}\nQe design = {external_q[index]:,}",
              transform=axis.transAxes, fontsize=8, color="#50565A")
    axis.set_xlabel("Bare-mode detuning (MHz)")
    if index == 0:
        axis.set_ylabel("Output / input (dB)")
fig.savefig("fridge_measurement_detail.svg")
plt.close(fig)
```

</details>

```{figure} ../images/fridge_measurement.svg
:alt: Three resonator features across 6–7 GHz, comparing the complete stationary mean with IQ samples at 1 ms and 10 ms integration.

Solid line: the complete stationary mean. Red points: one IQ sample per probe point.
Phase samples use the mean's unwrapped branch. [PDF](../images/fridge_measurement.pdf)
```

```{figure} ../images/fridge_measurement_detail.svg
:alt: Each of the three resonances resolved individually, with varied internal and design external Q values and 10 ms IQ samples.

The coupled resonance centers shift from the bare frequencies. All three
features use the same captured measurement. [PDF](../images/fridge_measurement_detail.pdf)
```

## Report noise at a defined plane

`noise_spectrum(readout)` reports normally ordered fluctuation density in
quanta at the receiver plane. Choose `unit="W/Hz"` or `unit="dBm/Hz"` for
available noise power density. Its final axis is the captured offset frequency;
the conversion uses the absolute sideband frequency. It excludes the coherent
signal and the detector vacuum added during IQ acquisition.

`statistics.noise_contributions(readout)` gives each source's contribution to
the **integrated IQ covariance**, in photons/ns. Its trace is the complex
field variance. The `device.correlations` term contains input-system
interference and may be negative; it is not an independent added noise source.
Source contributions, including `receiver.vacuum`, sum to the full covariance.

```python
statistics = measurement.statistics(receiver=IQReceiver(integration_time=1_000_000))
budget = statistics.noise_contributions(readout)
np.testing.assert_allclose(sum(budget.values()), statistics.covariance(readout), atol=1e-12)
noise_dbm_hz = measurement.noise_spectrum(readout, unit="dBm/Hz")
carrier_noise = noise_dbm_hz[..., len(measurement.noise_frequencies) // 2]
short_error = np.sqrt(np.mean(np.abs(short.ratio(readout)[0] - ideal)**2))
long_error = np.sqrt(np.mean(np.abs(long.ratio(readout)[0] - ideal)**2))
print(f"RESULT receiver_noise_dBm_per_Hz={np.mean(carrier_noise):.6f}")
print(f"RESULT short_complex_ratio_rmse={short_error:.6f}")
print(f"RESULT long_complex_ratio_rmse={long_error:.6f}")
```

Output:

```text
RESULT receiver_noise_dBm_per_Hz=-132.478062
RESULT short_complex_ratio_rmse=0.178530
RESULT long_complex_ratio_rmse=0.056456
```

The default offset grid spans ±0.1 GHz, logarithmically spaced down to 1 Hz.
Supply `noise_frequencies=` when other physical features need resolution.
Receiver processing checks integration convergence and edge support; it cannot
detect an unsampled feature. Declare frequency grids outside JAX transforms.
Physical parameter changes require a new capture. Receiver time, digital
transfer, calibration, and random draws operate on the captured arrays.
The samples describe stationary Gaussian second moments, including correlated
outputs; they do not generate continuous acquisition records or quantum trajectories.

For lifetime design, continue with [Purcell filtering and the T1 budget](slh-networks.md).
For pulse shaping and cavity depletion, see [pulses, leakage, and readout](dynamics-pulses-and-readout.md#empty-the-resonator-after-readout).
