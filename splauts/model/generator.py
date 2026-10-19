"""A shared MLP decodes each latent into bounded axis-aligned Gaussian splauts."""

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import Config


@dataclass
class Splauts:
    time: Tensor
    frequency: Tensor
    time_width: Tensor
    frequency_width: Tensor
    weight: Tensor

    def detach(self) -> "Splauts":
        return Splauts(**{f.name: getattr(self, f.name).detach() for f in fields(self)})


class SplautGenerator(nn.Module):
    def __init__(self, config: Config, hidden_dim: int = 64):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(nn.Linear(config.latent_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, config.splauts * 5))
        self.register_buffer("intensity_scale", torch.tensor(1.0))
        nn.init.normal_(self.network[-1].weight, std=0.01)
        with torch.no_grad():
            bias = self.network[-1].bias.view(config.splauts, 5)
            bias.zero_()
            bias[:, 0] = torch.linspace(-2, 2, config.splauts)
            bias[:, 1] = torch.linspace(-2, 2, config.splauts).roll(config.splauts // 3)
            bias[:, 2:4] = -1.5
            bias[:, 4] = -3.0

    def forward(self, latents: Tensor, durations: Tensor) -> Splauts:
        raw = self.network(latents).reshape(-1, self.config.splauts, 5)
        unit = raw.sigmoid()
        duration = durations.reshape(-1, 1)
        min_time = self.config.hop_length / self.config.sample_rate / 2
        max_time = torch.maximum(duration / 3, duration.new_tensor(min_time * 2))
        min_frequency = self.config.sample_rate / self.config.n_fft / 2
        max_frequency = self.config.sample_rate / 8
        return Splauts(unit[..., 0] * duration,
                       unit[..., 1] * self.config.sample_rate / 2,
                       min_time + unit[..., 2] * (max_time - min_time),
                       min_frequency + unit[..., 3] * (max_frequency - min_frequency),
                       F.softplus(raw[..., 4]) * self.intensity_scale)

    @torch.no_grad()
    def initialize_from_power(self, power: Tensor, times: Tensor, frequencies: Tensor,
                              duration: float, seed: int) -> None:
        """Seed shared output biases from training energy, never from station labels."""
        energy = power.sum(0).clamp_min(0)
        rng = torch.Generator().manual_seed(seed)
        indices = torch.multinomial(energy.flatten() + 1e-12, self.config.splauts,
                                    replacement=True, generator=rng)
        ti, fi = indices // len(frequencies), indices % len(frequencies)
        bias = self.network[-1].bias.view(self.config.splauts, 5)
        bias[:, 0] = torch.logit((times[ti] / duration).clamp(0.01, 0.99))
        bias[:, 1] = torch.logit((frequencies[fi] / (self.config.sample_rate / 2)).clamp(0.01, 0.99))
        bias[:, 2] = -1.3
        bias[:, 3] = -2.8
        bias[:, 4] = -3.0
        self.intensity_scale.copy_(power.max().clamp_min(1e-8))


def evaluate_field(splauts: Splauts, times: Tensor, frequencies: Tensor) -> Tensor:
    """Evaluate K fields on shared axes, returning [events, time, frequency]."""
    temporal = torch.exp(-0.5 * ((times[None, :, None] - splauts.time[:, None, :])
                                 / splauts.time_width[:, None, :]).square())
    spectral = torch.exp(-0.5 * ((frequencies[None, :, None] - splauts.frequency[:, None, :])
                                 / splauts.frequency_width[:, None, :]).square())
    return torch.einsum("ktn,kfn->ktf", temporal * splauts.weight[:, None, :], spectral)
