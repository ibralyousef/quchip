"""Physical stationary acquisition, independent of receiver configuration."""

from __future__ import annotations

from typing import Any

import numpy as np

from quchip.analysis.field_statistics import (
    block_diagonal, complex_covariance, field_transfer, proper_spectrum, quadrature_spectrum, quadrature_transfer,
)
from quchip.engine.output_network import output_mixing, pad_mixing
from quchip.engine.reference import (
    ReferenceFilter, ReferenceLoss, cw_transfer, noise_contributions, time_shift,
)
from quchip.results.measurement import MeasurementResult
from quchip.sweep import Sweep, ZippedSweep, _iter_axis_points
from quchip.utils.labeling import resolve_label


def capture_noise(operating: Any, backend: Any, labels: tuple[str, ...], frequency: Any, offsets: Any) -> Any:
    """Compute joint physical IQ spectra and source budgets from one operating point."""
    xp = backend.array_module
    engine = operating.engine
    rho = xp.asarray(backend.to_array(operating.state.state))
    indices = {channel.key: i for i, channel in enumerate(engine.slh.channels)}
    selected = [indices[label] for label in labels]
    channels = [engine.slh.channels[i] for i in selected]
    mixed = engine.slh.output_network is not None
    regression_labels = tuple(channel.key for channel in engine.slh.channels) if mixed else labels
    excess = quadrature_spectrum(engine, rho, backend, operating.prepared, regression_labels, offsets)
    upper = xp.stack([cw_transfer(c.reference.outbound, frequency + offsets, xp) + xp.zeros_like(offsets)
                      for c in channels], axis=-1)
    lower = xp.stack([cw_transfer(c.reference.outbound, frequency - offsets, xp) + xp.zeros_like(offsets)
                      for c in channels], axis=-1)
    if mixed:
        graph = engine.slh.output_network
        graph_positive = [graph.evaluate(frequency + offset, xp) for offset in offsets]
        graph_negative = [graph.evaluate(frequency - offset, xp) for offset in offsets]
        graph_upper = xp.stack([pad_mixing(point[0], len(engine.slh.channels), xp)[xp.asarray(selected)]
                                for point in graph_positive])
        graph_lower = xp.stack([pad_mixing(point[0], len(engine.slh.channels), xp)[xp.asarray(selected)]
                                for point in graph_negative])
        transform = field_transfer(upper[..., None] * graph_upper, lower[..., None] * graph_lower, xp)
    else:
        transform = block_diagonal(quadrature_transfer(upper, lower, xp), xp)
    transformed = transform @ excess @ xp.conj(xp.swapaxes(transform, -1, -2))
    size = 2 * len(labels)
    zero = xp.zeros((size, size))
    empty = xp.zeros((len(offsets), size, size), dtype=complex)
    components: dict[str, tuple[Any, Any]] = {"device.correlations": (zero, transformed)}
    gains = xp.stack([cw_transfer(c.reference.outbound, frequency, xp) for c in channels])
    if mixed:
        dc = gains[:, None] * output_mixing(engine.slh, frequency, xp)[xp.asarray(selected)]
        dc_transform = field_transfer(dc, dc, xp)
    else:
        dc_transform = block_diagonal(quadrature_transfer(gains, gains, xp), xp)
    has_filter = any(isinstance(element, ReferenceFilter) for c in channels for element in c.reference.outbound)
    scattering = xp.asarray(engine.slh.S)
    if not mixed:
        scattering = scattering[xp.asarray(selected)]
    has_filter = has_filter or (engine.slh.output_network is not None and engine.slh.output_network.colored)
    for index, channel in enumerate(engine.slh.channels):
        column = scattering[:, index]
        source_matrix = column[:, None] * xp.conj(column[None, :])
        inbound = channel.reference.inbound
        reference_sources = any(
            (isinstance(element, ReferenceLoss) and element.occupation is not None)
            or (isinstance(element, ReferenceFilter) and element.loss_occupation is not None)
            for element in inbound)
        if reference_sources:
            positive = noise_contributions(inbound, frequency + offsets, xp)
            negative = noise_contributions(inbound, frequency - offsets, xp)
            colored = has_filter or any(isinstance(element, ReferenceFilter) for element in inbound)
            for name, up in positive.items():
                direct = proper_spectrum(up[..., None, None] * source_matrix,
                                         negative[name][..., None, None] * source_matrix, xp)
                spectrum = transform @ direct @ xp.conj(xp.swapaxes(transform, -1, -2))
                components[f"input.{channel.key}.{name}"] = ((zero, spectrum) if colored else
                    (xp.real(dc_transform @ direct[len(offsets)//2] @ xp.conj(dc_transform.T)), empty))
        elif channel.input_occupation is not None:
            direct = complex_covariance(source_matrix * channel.input_occupation, xp)
            if has_filter:
                components[f"input.{channel.key}"] = (
                    zero, transform @ direct @ xp.conj(xp.swapaxes(transform, -1, -2)))
            else:
                components[f"input.{channel.key}"] = (xp.real(dc_transform @ direct @ xp.conj(dc_transform.T)), empty)
    if engine.slh.output_network is not None:
        for name in graph_positive[0][1]:
            up = xp.stack([point[1][name][xp.asarray(selected)[:, None], xp.asarray(selected)[None, :]]
                           for point in graph_positive]) * upper[..., :, None] * xp.conj(upper[..., None, :])
            down = xp.stack([point[1][name][xp.asarray(selected)[:, None], xp.asarray(selected)[None, :]]
                             for point in graph_negative]) * lower[..., :, None] * xp.conj(lower[..., None, :])
            spectrum = proper_spectrum(up, down, xp)
            components[f"network.{name}"] = ((zero, spectrum) if has_filter else
                                              (xp.real(spectrum[len(offsets)//2]), empty))
    for index, channel in enumerate(channels):
        elements = channel.reference.outbound
        positive = noise_contributions(elements, frequency + offsets, xp)
        negative = noise_contributions(elements, frequency - offsets, xp)
        for name, up in positive.items():
            down = negative[name]
            even, odd = (up + down) / 4, (up - down) / 4
            block = xp.stack((xp.stack((even, 1j * odd), axis=-1),
                              xp.stack((-1j * odd, even), axis=-1)), axis=-2)
            selector = xp.eye(len(labels))[index:index+1].T @ xp.eye(len(labels))[index:index+1]
            spectrum = xp.kron(selector, block)
            position = next(i for i, element in enumerate(elements) if element.label == name)
            colored = any(isinstance(element, ReferenceFilter) for element in elements[position:])
            if colored:
                components[f"output.{channel.key}.{name}"] = (zero, spectrum)
            else:
                components[f"output.{channel.key}.{name}"] = (xp.real(spectrum[0]), empty)
    return components, xp.asarray([time_shift(c.reference.outbound) for c in channels])


def measure(
    vna: Any, frequencies: Any, amplitudes: Any, variations: tuple[Sweep | ZippedSweep, ...], *,
    input: Any, outputs: Any, noise_frequencies: Any, options: Any, progress: bool,
) -> MeasurementResult:
    """Acquire physical fields and correlations once per probe/sweep point."""
    from quchip.analysis.vna import (
        _axis_values, _operating_point, _plane_means, _public_axes, _resolve_exposure,
    )
    if isinstance(outputs, str):
        raise TypeError("outputs must be a sequence of plane objects or labels, not a string.")
    labels = tuple(vna.ports) if outputs is None else tuple(_resolve_exposure(vna.chip, plane) for plane in outputs)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("outputs must contain distinct exposed planes.")
    if input is None:
        if len(vna.ports) != 1:
            raise ValueError("measure() requires input= when more than one port is selected.")
        input_label = vna.ports[0]
    else:
        input_label = _resolve_exposure(vna.chip, input)
    if any(tone.port == input_label for tone in vna._tones):
        raise ValueError("The measurement probe must be the only tone on its input.")
    frequency_values, frequency_axis = _axis_values(frequencies)
    amplitude_values, amplitude_axis = _axis_values(amplitudes)
    vna._validate_variations(variations, reserved=(
        *(("frequency",) if frequency_axis else ()), *(("amplitude",) if amplitude_axis else ())))
    shape, variation_points = _iter_axis_points(variations)
    axes = _public_axes(variations)
    if amplitude_axis:
        shape += (len(amplitude_values),)
        axes += (("amplitude", amplitude_values),)
    if frequency_axis:
        shape += (len(frequency_values),)
        axes += (("frequency", frequency_values),)
    if noise_frequencies is None:
        positive = np.geomspace(1e-9, 0.1, 161)
        offsets = np.concatenate((-positive[::-1], [0.0], positive))
    else:
        offsets = np.asarray(noise_frequencies, dtype=float)
    if (offsets.ndim != 1 or len(offsets) < 5 or not np.all(np.isfinite(offsets))
            or np.any(np.diff(offsets) <= 0) or not np.any(offsets == 0)
            or not np.allclose(offsets, -offsets[::-1], atol=0, rtol=1e-12)):
        raise ValueError("noise_frequencies must be finite, increasing, symmetric offsets including zero (GHz).")
    xp = vna.chip.backend.array_module
    captured, means, incident, diagnostics, delays, parameters = [], [], [], [], [], []
    points = [(params, amplitude, frequency) for _, params in variation_points
              for amplitude in amplitude_values for frequency in frequency_values]
    if progress:
        from tqdm import tqdm
        points = tqdm(points, desc="VNA measurement")
    for params, amplitude, frequency in points:
        chip = vna._chip_at(params)
        parameters.append(dict(chip.parameters))
        tones = vna._tone_values(params) + ((input_label, frequency, amplitude),)
        operating = _operating_point(chip, tones, frequency, (), options)
        means.append(_plane_means(operating.engine, operating.state.state, chip.backend, labels, tones, frequency))
        components, output_delays = capture_noise(operating, chip.backend, labels, frequency, xp.asarray(offsets))
        captured.append(components)
        delays.append(output_delays)
        incident.append(xp.asarray(amplitude))
        diagnostics.append(operating.diagnostics)
    size = 2 * len(labels)
    names = tuple(captured[0])
    if any(tuple(point) != names for point in captured):
        raise ValueError("Measurement variations must preserve noise source structure.")
    components = {
        name: (xp.stack([point[name][0] for point in captured]).reshape((*shape, size, size)),
               xp.stack([point[name][1] for point in captured]).reshape((*shape, len(offsets), size, size)))
        for name in names
    }
    return MeasurementResult(
        ports=labels, input=resolve_label(input_label),
        frequencies=frequency_values if frequency_axis else frequency_values[0],
        amplitudes=amplitude_values if amplitude_axis else amplitude_values[0], axes=axes, shape=shape,
        diagnostics=tuple(diagnostics), values=xp.stack(means).reshape((*shape, len(labels))),
        incident=xp.stack(incident).reshape(shape), noise_frequencies=offsets, noise_components=components,
        output_delays=xp.stack(delays).reshape((*shape, len(labels))), parameters=tuple(parameters),
    )
