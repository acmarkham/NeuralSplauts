"""Configurable spectral absorption law, separate from geometric spreading.

Loss in dB = alpha_1kHz_dB_per_km * distance_km * (frequency_kHz)**p.
This illustrative power law is not a calibrated weather-dependent atmosphere.
"""

import math

from torch import Tensor

from .config import Config


def absorption_transmission(distances: Tensor, frequencies: Tensor, config: Config,
                            *, power: bool = False) -> Tensor:
    """Return [..., frequency] amplitude or power transmission, with position gradients."""
    loss_db = (config.absorption_db_per_km * distances[..., None] / 1000
               * (frequencies / 1000).pow(config.absorption_power))
    return (-math.log(10) * loss_db / (10 if power else 20)).exp()
