"""One geographic position and onset anchor per event, shared by all splauts."""

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn


@dataclass
class EventState:
    positions: Tensor
    onsets: Tensor
    amplitudes: Tensor
    durations: Tensor
    latents: Tensor

    def detach(self) -> "EventState":
        return EventState(**{f.name: getattr(self, f.name).detach().clone() for f in fields(self)})


class EventParameters(nn.Module):
    def __init__(self, initial: EventState, area_size: float, max_onset: float,
                 optimize_onsets: bool = False, optimize_amplitudes: bool = False,
                 optimize_latents: bool = False):
        super().__init__()
        count = len(initial.positions)
        if initial.positions.shape != (count, 2) or count == 0:
            raise ValueError("Positions must have shape [events, 2]")
        if any(value.shape != (count,) for value in (initial.onsets, initial.amplitudes, initial.durations)):
            raise ValueError("Onsets, amplitudes and durations must have one value per event")
        if initial.latents.ndim != 2 or len(initial.latents) != count:
            raise ValueError("Latents must have shape [events, latent_dim]")
        if any(not torch.isfinite(getattr(initial, f.name)).all() for f in fields(initial)):
            raise ValueError("Event values must be finite")
        if (initial.amplitudes <= 0).any() or (initial.durations <= 0).any():
            raise ValueError("Amplitudes and durations must be positive")
        if ((initial.positions < 0) | (initial.positions > area_size)).any():
            raise ValueError("Initial positions lie outside the observation area")
        if ((initial.onsets < 0) | (initial.onsets > max_onset)).any():
            raise ValueError("Initial onsets lie outside the recording")
        self.area_size, self.max_onset = area_size, max_onset
        self.position_logits = nn.Parameter(torch.logit((initial.positions / area_size).clamp(1e-5, 1-1e-5)))
        self.onset_logits = nn.Parameter(torch.logit((initial.onsets / max_onset).clamp(1e-5, 1-1e-5)), requires_grad=optimize_onsets)
        self.log_amplitudes = nn.Parameter(initial.amplitudes.log().clone(), requires_grad=optimize_amplitudes)
        self.latents = nn.Parameter(initial.latents.clone(), requires_grad=optimize_latents)
        self.register_buffer("durations", initial.durations.clone())

    def forward(self) -> EventState:
        return EventState(self.position_logits.sigmoid() * self.area_size,
                          self.onset_logits.sigmoid() * self.max_onset,
                          self.log_amplitudes.exp(), self.durations, self.latents)
