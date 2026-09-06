# Resonator readout and fridge wiring

Measure a one-port resonator's reflection and its source-to-receiver response
through the fridge. Frequencies are in GHz, times in ns, and decay rates
in `1/ns`. Run the cells in order.

## Measure the resonator reflection

The resonator has internal loss Q = 200,000 and one external coupling port
with κext/2π = 1 MHz. `VNA.sweep()` returns the small-signal S parameters at
the exposed ports.

```python
import numpy as np
from quchip import Chip, PortNetwork, Resonator, VNA

r = Resonator(freq=7.0, levels=3, internal_quality_factor=200_000, label="r")
network = PortNetwork(label="chip_port")
coupler = network.port("coupler", target=r, rate=2 * np.pi * 0.001)
chip_port = network.expose("chip", at=coupler)
chip = Chip([r], port_network=network, frame="rotating")
```

Sweep the probe frequency at the exposed chip port.

```python
frequencies = np.linspace(6.995, 7.005, 401)
vna = VNA(chip, ports=[chip_port])
s11 = vna.sweep(frequencies).s(chip_port, chip_port)
kappa = coupler.rate + 2 * np.pi * r.freq / r.internal_quality_factor
```

<details>
<summary>Plot S11</summary>

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "legend.fontsize": 9, "lines.linewidth": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42,
})
fig, magnitude_axis = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
phase_axis = magnitude_axis.twinx()
magnitude_axis.plot(frequencies, 20 * np.log10(np.abs(s11)), color="#C92F33")
phase_axis.plot(frequencies, np.unwrap(np.angle(s11)) * 180 / np.pi, color="#246FA8", ls="--")
magnitude_axis.set(xlabel="Probe frequency (GHz)", ylabel=r"$|S_{11}|$ (dB)",
                   xlim=(frequencies[0], frequencies[-1]))
phase_axis.set(ylabel=r"Phase of $S_{11}$ (degrees)", yticks=[0, 90, 180, 270, 360])
magnitude_axis.yaxis.label.set_color("#C92F33")
magnitude_axis.tick_params(axis="y", colors="#C92F33")
phase_axis.yaxis.label.set_color("#246FA8")
phase_axis.tick_params(axis="y", colors="#246FA8")
phase_axis.spines["right"].set_visible(True)
phase_axis.spines["right"].set_color("#246FA8")
magnitude_axis.spines["left"].set_color("#C92F33")
magnitude_axis.ticklabel_format(useOffset=False, axis="x")
magnitude_axis.axvline(r.freq, color="0.8", lw=0.7, ls=":", zorder=0)
fig.savefig("resonator_s11.svg")
fig.savefig("resonator_s11.pdf")
plt.close(fig)
```

</details>

```{figure} ../images/resonator_s11.svg
:alt: One-port resonator reflection magnitude and phase through the 7 GHz resonance.

Reflection at the chip port: a 1.035 MHz loaded linewidth and −0.608 dB at
resonance. The shallow dip and full phase winding identify an overcoupled resonator. [PDF](../images/resonator_s11.pdf)
```

For vacuum input and a harmonic resonator,

```{math}
S_{11}(f)=1-\frac{\kappa_{\rm ext}}{\kappa/2-2\pi i(f-f_r)},
\qquad \kappa=\kappa_{\rm int}+\kappa_{\rm ext}.
```

```python
expected = 1 - coupler.rate / (kappa / 2 - 2j * np.pi * (frequencies - r.freq))
np.testing.assert_allclose(s11, expected, rtol=1e-10, atol=1e-12)
```

## Put the chip in the fridge

A mixing-chamber circulator sends port 1 to the chip on port 2 and routes
its reflection to port 3. The chip still has one coupling port. Input
attenuation is distributed across 4 K, the cold plate, and the mixing chamber;
two output isolators precede the 4 K HEMT.

```python
fridge = PortNetwork(label="fridge")
input_delay = fridge.delay("input_cable", duration=50.0)
output_delay = fridge.delay("output_cable", duration=50.0)
port = fridge.port("coupler", target=r, rate=2 * np.pi * 0.001)
att_4k = fridge.attenuator("att_4K", eta=0.01)
att_cp = fridge.attenuator("att_CP", eta=0.01)
att_mxc = fridge.attenuator("att_MXC", eta=0.01)
ir = fridge.attenuator("ir_passband", eta=10**(-1 / 10))
circ = fridge.circulator("circ")
arm = fridge.attenuator("circ_insertion", eta=10**(-0.5 / 10))
iso_1 = fridge.isolator("iso_1")
iso_2 = fridge.isolator("iso_2")
iso_loss = fridge.attenuator("iso_insertion", eta=10**(-1 / 10))
coax = fridge.attenuator("output_coax", eta=10**(-1 / 10))
hemt = fridge.amplifier("HEMT_4K", gain=1e4, added_noise=12.0)
room_amp = fridge.amplifier("amp_RT", gain=1e2, added_noise=600.0)
```

`added_noise` specifies input-referred amplifier noise in quanta (symmetrized).
It raises the receiver noise floor without changing the coherent S parameters.
The stage temperatures below label the wiring; they do not set thermal baths.

Connect the input chain to circulator side 1, the chip to side 2, and the
receiver chain to side 3. Expose `p1` for the source and `p2` for the receiver.

```python
fridge.link(input_delay, att_4k, att_cp, att_mxc, ir, circ.side(1))
fridge.link(circ.side(2), arm, port)
fridge.link(circ.side(3), iso_1, iso_loss, iso_2, coax, hemt, room_amp, output_delay)
p1 = fridge.expose("p1", at=input_delay.side(1))
p2 = fridge.expose("p2", at=output_delay.side(2))
readout_chip = Chip([r], port_network=fridge, frame="rotating")
```

<details>
<summary>Draw the fridge wiring</summary>

```python
from matplotlib.patches import Circle, FancyBboxPatch

fig, axis = plt.subplots(figsize=(7.2, 6.5), layout="constrained")
axis.set(xlim=(0, 9.7), ylim=(-0.1, 10.2))
axis.set_aspect("equal")
axis.axis("off")
blue, red, ink = "#246FA8", "#C92F33", "#16181C"
for label, bottom, top, fill in [
    ("300 K", 8.65, 10.1, "#ffffff"),
    ("50 K", 7.5, 8.65, "#f2f5f7"),
    ("4 K", 6.15, 7.5, "#eaf0f5"),
    ("Still\n800 mK", 4.85, 6.15, "#f2f5f7"),
    ("Cold plate\n100 mK", 3.5, 4.85, "#eaf0f5"),
    ("Mixing\nchamber\n20 mK", 0.0, 3.5, "#e0ebf2"),
]:
    axis.axhspan(bottom, top, color=fill, zorder=0)
    axis.text(0.15, (bottom + top) / 2, label, va="center", fontsize=9, color=ink)
    axis.hlines(bottom, 0, 9.7, color="#cbd5dd", linewidth=0.7)

# Input descends; output rises. The chip arm is one bidirectional cable.
axis.plot([2.5, 2.5, 4.75], [9.25, 2.2, 2.2], color=blue, lw=2)
axis.plot([5.55, 8.15, 8.15], [2.2, 2.2, 9.8], color=red, lw=2)
axis.annotate("", (5.15, 0.95), (5.15, 1.8), arrowprops={"arrowstyle": "<->", "color": ink, "lw": 1.6})
for x, direction, color in [(2.5, -1, blue), (8.15, 1, red)]:
    for y in [8.55, 6.0, 4.75]:
        axis.annotate("", (x, y + direction * 0.35), (x, y), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4})
    for y in [8.05, 5.5]:
        axis.plot(x, y, "o", color=color, ms=4)
axis.text(3.0, 8.0, "Coax anchored\nat each stage\n50 ns per line", fontsize=8, color="#536574", va="center")

for x, y, width, height, label, color in [
    (2.5, 9.25, 1.65, 0.58, "Source · p1", blue),
    (2.5, 6.8, 1.45, 0.62, "20 dB", blue),
    (2.5, 4.2, 1.45, 0.62, "20 dB", blue),
    (2.5, 2.2, 1.20, 0.62, "20 dB", blue),
    (3.9, 2.2, 1.00, 0.62, "IR filter\n1 dB", blue),
    (6.45, 2.2, 1.12, 0.62, "Isolator\n0.5 dB", red),
    (8.15, 2.2, 1.12, 0.62, "Isolator\n0.5 dB", red),
    (8.15, 4.2, 1.50, 0.62, "Output coax\n1 dB", red),
    (8.15, 6.8, 1.50, 0.62, "HEMT\n+40 dB", red),
    (8.15, 9.0, 1.50, 0.62, "Amplifier\n+20 dB", red),
    (5.15, 0.55, 2.4, 0.78, "Resonator chip\n7 GHz · one port", ink),
]:
    axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                   boxstyle="round,pad=0.04,rounding_size=0.06", ec=color, fc="white", lw=1.1, zorder=3))
    axis.text(x, y, label, ha="center", va="center", fontsize=8.5, color=ink, zorder=4)
axis.add_patch(Circle((5.15, 2.2), 0.4, ec=ink, fc="white", lw=1.3, zorder=3))
axis.text(5.15, 2.2, "↺", ha="center", va="center", fontsize=22, color=ink)
for x, y, label in [(4.65, 2.52, "1"), (5.42, 1.52, "2"), (5.7, 2.52, "3")]:
    axis.text(x, y, label, ha="center", fontsize=8, color=ink)
axis.text(5.15, 3.05, "Circulator\n0.5 dB per pass", ha="center", va="center", fontsize=8, color=ink)
axis.text(8.15, 10.0, "Receiver · p2", ha="center", fontsize=9, color=red)
axis.text(6.25, 1.25, "Reverse fields →\nmatched isolator loads", fontsize=8, color="#536574")
fig.savefig("fridge_wiring.svg")
fig.savefig("fridge_wiring.pdf")
plt.close(fig)
```

</details>

```{figure} ../images/fridge_wiring.svg
:alt: Fridge stages with 20 dB attenuators at 4 K, 100 mK and 20 mK; a mixing-chamber circulator connects the one-port resonator to two output isolators, a 4 K HEMT and a room-temperature amplifier.

The blue input line reaches the chip through circulator ports 1 → 2. The
reflection returns through 2 → 3 into the red output line. Loss labels are
power attenuation per traversal. [PDF](../images/fridge_wiring.pdf)
```

The diagram labels power gains and losses; 20 dB attenuation means `eta=0.01`.
Values are illustrative; the stage arrangement follows
[Krinner et al.](https://arxiv.org/abs/1806.07862).

`link()` joins physical sides; `expose()` names instrument ports. The
circulator routes 1 → 2 → 3 → 1, while isolators terminate reverse fields.

## Measure the reflection through the circulator

The VNA has two instrument ports: source 1 and receiver 2. Their transmission
is S21, even though the receiver connects to side 3 of the circulator.

```python
fridge_vna = VNA(readout_chip, ports=[p1, p2])
response = fridge_vna.sweep(frequencies)
s21 = response.s(p2, p1)
line_db = -60 - 1 - 2 * 0.5 - 1 - 1 + 40 + 20
cable_phase = np.exp(2j * np.pi * frequencies * 100.0)
np.testing.assert_allclose(s21, 10**(line_db / 20) * cable_phase * s11, rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(response.s(p1, p2), 0.0, atol=1e-12)
```

<details>
<summary>Plot S21</summary>

```python
fig, magnitude_axis = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
phase_axis = magnitude_axis.twinx()
magnitude_axis.plot(frequencies, 20 * np.log10(np.abs(s21)), color="#C92F33")
phase_axis.plot(frequencies, np.unwrap(np.angle(s21)) * 180 / np.pi, color="#246FA8", ls="--")
magnitude_axis.set(xlabel="Probe frequency (GHz)", ylabel=r"$|S_{21}|$ (dB)",
                   xlim=(frequencies[0], frequencies[-1]))
phase_axis.set_ylabel(r"Phase of $S_{21}$ (degrees)")
magnitude_axis.yaxis.label.set_color("#C92F33")
magnitude_axis.tick_params(axis="y", colors="#C92F33")
phase_axis.yaxis.label.set_color("#246FA8")
phase_axis.tick_params(axis="y", colors="#246FA8")
phase_axis.spines["right"].set_visible(True)
phase_axis.spines["right"].set_color("#246FA8")
magnitude_axis.spines["left"].set_color("#C92F33")
magnitude_axis.ticklabel_format(useOffset=False, axis="x")
magnitude_axis.axvline(r.freq, color="0.8", lw=0.7, ls=":", zorder=0)
magnitude_axis.axhline(line_db, color="0.6", ls=":", lw=0.8)
fig.savefig("fridge_s21.svg")
fig.savefig("fridge_s21.pdf")
plt.close(fig)
```

</details>

```{figure} ../images/fridge_s21.svg
:alt: Direct source-to-receiver S21 magnitude and phase for the fridge reflection setup.

S21 at the room-temperature instrument ports, including 100 ns total cable
delay. The gains and losses give a −4 dB background; resonance lowers it to
−4.608 dB. [PDF](../images/fridge_s21.pdf)
```

The ideal circulator routes fields with unit phase; it does not introduce an
extra reflection. S21 contains the resonator phase **plus** the phase accumulated
along both cable legs. Here each leg has 50 ns delay. quchip uses
$e^{-i2\pi ft}$ fields, so the delay factor is $e^{+i2\pi f(100\,\mathrm{ns})}$.
These reference delays affect the reported field, not the chip dynamics.

These curves are the coherent mean response, not a noisy measurement trace.
`fridge_vna.output_spectrum(p2, frequencies=offsets)` reports the receiver
fluctuation spectrum, including amplifier noise; `offsets` are in GHz from the
stationary frame. Predicting scatter in a measured S21 trace also requires the
probe power and receiver bandwidth or averaging time.

The reverse path is zero for the ideal isolators. This model omits amplifier
saturation, finite reverse isolation, and thermal excitation.

For lifetime design, continue with [Purcell filtering and the T1 budget](slh-networks.md).
For pulse shaping and cavity depletion, see [pulses, leakage, and readout](dynamics-pulses-and-readout.md#empty-the-resonator-after-readout).
