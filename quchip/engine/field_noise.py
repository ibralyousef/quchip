"""Thermal input fields in the instantaneous network's physical channel basis."""

from typing import Any, Mapping

import numpy as np

from quchip.utils.constants import k_B
from quchip.utils.jax_utils import contains_tracer, select_array_module


def noise_parameters(*, occupation: Any = None, temperature: Any = None, noise_frequency: Any = None) -> dict[str, Any]:
    """Validate one Markov field state; temperature is mK and frequency is GHz."""
    if temperature is not None:
        if occupation is not None or noise_frequency is None:
            raise ValueError("Specify occupation, or temperature with noise_frequency in GHz.")
        values = {"temperature": temperature, "noise_frequency": noise_frequency}
    elif occupation is not None:
        if noise_frequency is not None:
            raise ValueError("noise_frequency is used only with temperature.")
        values = {"occupation": occupation}
    elif noise_frequency is not None:
        raise ValueError("noise_frequency requires temperature.")
    else:
        return {}
    for key, value in values.items():
        if not contains_tracer(value):
            number = np.asarray(value)
            if number.ndim or not np.isrealobj(number) or not np.isfinite(number):
                raise ValueError(f"{key} must be a finite real scalar.")
            if number < 0 or (key == "noise_frequency" and number == 0):
                raise ValueError(f"{key} must be {'positive' if key == 'noise_frequency' else 'non-negative'}.")
    return values


def occupation_value(parameters: Mapping[str, Any]) -> Any | None:
    """Return a declared input occupation, or None for an implicit vacuum field."""
    values = noise_parameters(**{key: parameters[key] for key in
                                ("occupation", "temperature", "noise_frequency") if key in parameters})
    if not values:
        return None
    if "occupation" in values:
        return values["occupation"]
    xp = select_array_module(contains_tracer(tuple(values.values())))
    temperature = xp.asarray(values["temperature"])
    safe_temperature = xp.where(temperature > 0, temperature, 1.0)
    exponent = values["noise_frequency"] / (k_B * safe_temperature)
    # exp(-x)/(1-exp(-x)) is stable at low temperature and at T=0 under JAX.
    occupation = xp.exp(-exponent) / (-xp.expm1(-exponent))
    return xp.where(temperature > 0, occupation, 0.0)


def input_noise_matrix(slh: Any, xp: Any) -> Any:
    """Return normally ordered white input noise routed to the output basis."""
    occupations = xp.asarray([0.0 if c.input_occupation is None else c.input_occupation for c in slh.channels])
    scattering = xp.asarray(slh.S)
    return (scattering * occupations[None, :]) @ xp.conj(scattering.T)
