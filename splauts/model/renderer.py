"""Continuous travel times and power attenuation; no frame-quantized delays."""

import torch
from torch import Tensor

from ..config import Config
from ..propagation import absorption_transmission
from .generator import Splauts, gaussian_kernel


def render(splauts: Splauts, positions: Tensor, onsets: Tensor, amplitudes: Tensor,
           stations: Tensor, times: Tensor, frequencies: Tensor, config: Config) -> Tensor:
    """Render [stations, time, frequency]; amplitudes are source POWER scales.

    Stations may be subsets here for diagnostics; primary simulation and
    localization entry points enforce the four-station geometry requirement.
    """
    # Tiny regularization defines the derivative even at station/source coincidence.
    distances = ((stations[:, None, :] - positions[None, :, :]).square().sum(-1) + 1e-12).sqrt()
    relative_times = times[None, None, :, None] - onsets[None, :, None, None] - distances[:, :, None, None] / config.sound_speed
    temporal = gaussian_kernel((relative_times - splauts.time[None, :, None, :])
                                / splauts.time_width[None, :, None, :])
    spectral = gaussian_kernel((frequencies[None, :, None] - splauts.frequency[:, None, :])
                                / splauts.frequency_width[:, None, :])
    gains = amplitudes[None, :] / (distances + config.distance_offset).pow(config.power_spreading_exponent)
    weighted = temporal * splauts.weight[None, :, None, :] * gains[:, :, None, None]
    if config.absorption_db_per_km:
        emitted = torch.einsum("mktn,kfn->mktf", weighted, spectral)
        transmission = absorption_transmission(distances, frequencies, config, power=True)
        return (emitted * transmission[:, :, None, :]).sum(1)
    return torch.einsum("mktn,kfn->mtf", weighted, spectral)
