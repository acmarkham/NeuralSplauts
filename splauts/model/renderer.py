"""Continuous travel times and power attenuation; no frame-quantized delays."""

import torch
from torch import Tensor

from ..config import Config
from .generator import Splauts


def render(splauts: Splauts, positions: Tensor, onsets: Tensor, amplitudes: Tensor,
           stations: Tensor, times: Tensor, frequencies: Tensor, config: Config) -> Tensor:
    """Render [stations, time, frequency]; amplitudes are source POWER scales.

    Stations may be subsets here for diagnostics; primary simulation and
    localization entry points enforce the four-station geometry requirement.
    """
    # Tiny regularization defines the derivative even at station/source coincidence.
    distances = ((stations[:, None, :] - positions[None, :, :]).square().sum(-1) + 1e-12).sqrt()
    relative_times = times[None, None, :, None] - onsets[None, :, None, None] - distances[:, :, None, None] / config.sound_speed
    temporal = torch.exp(-0.5 * ((relative_times - splauts.time[None, :, None, :])
                                 / splauts.time_width[None, :, None, :]).square())
    spectral = torch.exp(-0.5 * ((frequencies[None, :, None] - splauts.frequency[:, None, :])
                                 / splauts.frequency_width[:, None, :]).square())
    gains = amplitudes[None, :] / (distances + config.distance_offset).square()
    weighted = temporal * splauts.weight[None, :, None, :] * gains[:, :, None, None]
    return torch.einsum("mktn,kfn->mtf", weighted, spectral)
