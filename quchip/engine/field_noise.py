"""Thermal input fields in the instantaneous network's physical channel basis."""

from typing import Any, Mapping

import numpy as np

from quchip.utils.jax_utils import contains_tracer


def _choice(parameters: Mapping[str, Any], names: tuple[str, ...]) -> tuple[str, Any]:
    """Select one explicit numerical convention, retaining traced values."""
    values = [(name, parameters[name]) for name in names if parameters.get(name) is not None]
    if len(values) != 1:
        raise ValueError(f"Specify exactly one of {', '.join(names)}.")
    name, value = values[0]
    if not contains_tracer(value):
        number = np.asarray(value)
        if number.ndim or not np.isrealobj(number) or not np.isfinite(number):
            raise ValueError(f"{name} must be a finite real scalar.")
    return name, value


def attenuation_value(parameters: Mapping[str, Any]) -> Any:
    """Convert authored power transmission or positive dB loss to transmission."""
    name, value = _choice(parameters, ("eta", "loss_db"))
    eta = value if name == "eta" else 10 ** (-value / 10)
    if not contains_tracer(eta) and not 0 <= eta <= 1:
        raise ValueError("Attenuator eta must lie in [0, 1]; loss_db must be non-negative.")
    return eta


def amplifier_values(parameters: Mapping[str, Any]) -> tuple[Any, Any]:
    """Resolve power gain and input-referred symmetrized added quanta."""
    name, value = _choice(parameters, ("gain", "gain_db"))
    gain = value if name == "gain" else 10 ** (value / 10)
    _, noise = _choice(parameters, ("added_noise",))
    return gain, noise


def noise_parameters(thermal_occupation: Any = None) -> dict[str, Any]:
    """Validate a constant thermal population in quanta; None means implicit vacuum."""
    values = {} if thermal_occupation is None else {"thermal_occupation": thermal_occupation}
    thermal_occupation_value(values)
    return values


def thermal_occupation_value(parameters: Mapping[str, Any]) -> Any | None:
    """Return the declared thermal quanta, preserving an explicit zero."""
    if parameters.get("thermal_occupation") is None:
        return None
    _, value = _choice(parameters, ("thermal_occupation",))
    if not contains_tracer(value) and value < 0:
        raise ValueError("thermal_occupation must be non-negative.")
    return value


def input_noise_matrix(slh: Any, xp: Any) -> Any:
    """Return normally ordered white input noise routed to the output basis."""
    occupations = xp.asarray([0.0 if c.input_occupation is None else c.input_occupation for c in slh.channels])
    scattering = xp.asarray(slh.S)
    return (scattering * occupations[None, :]) @ xp.conj(scattering.T)
