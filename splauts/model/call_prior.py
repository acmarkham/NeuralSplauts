"""An acoustic latent domain for a supervised, transferable splaut decoder.

Frequency translation is explicit; the MLP learns relative time/frequency
structure. The family and acoustic coordinates are known during pretraining and
must be inferred from observations during mixture inversion.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from ..config import Config
from .generator import SplautGenerator, Splauts

# Frequency (Hz), bandwidth (Hz), modulation rate (Hz), pulse rate (Hz).
ACOUSTIC_RANGES = ((800., 2500.), (100., 500.), (2., 8.), (12., 20.))


def acoustic_ranges(config: Config = Config()) -> tuple:
    return tuple((lo * config.frequency_scale, hi * config.frequency_scale) if i < 2 else (lo, hi)
                 for i, (lo, hi) in enumerate(ACOUSTIC_RANGES))


def acoustic_values(codes: Tensor, config: Config = Config()) -> Tensor:
    lower = codes.new_tensor([r[0] for r in acoustic_ranges(config)])
    upper = codes.new_tensor([r[1] for r in acoustic_ranges(config)])
    start = len(config.call_families)
    return lower + (codes[:, start:start+4] + 1) * 0.5 * (upper - lower)


def codes_from_logits(families: Tensor, logits: Tensor, latent_dim: int, config: Config = Config()) -> Tensor:
    count = len(config.call_families)
    return torch.cat((F.one_hot(families, count).to(logits.dtype), 2 * logits.sigmoid() - 1,
                      logits.new_zeros(len(logits), latent_dim - count - 4)), dim=-1)


def logits_from_codes(codes: Tensor, config: Config = Config()) -> Tensor:
    start = len(config.call_families)
    return torch.logit(((codes[:, start:start+4] + 1) / 2).clamp(1e-5, 1-1e-5))


class CallPriorGenerator(SplautGenerator):
    """Shared MLP and Gaussian field, with explicit harmonic frequency anchors.

    Latents contain a family code and four bounded acoustic values.
    Remaining dimensions are zero. Absolute frequency bypasses the MLP, so it
    can transfer a learned shape to frequencies absent from the training set.
    Splauts are interleaved across harmonic orders within each event; all bands
    retain the event's single code, location, onset and power scale.
    """

    def __init__(self, config: Config):
        if config.latent_dim < len(config.call_families) + 4 or config.sample_rate < 6000:
            raise ValueError("The call prior needs one latent per family plus four acoustic coordinates, and sample_rate >=6000")
        if config.splauts < config.harmonic_count:
            raise ValueError("Supply at least one splaut per harmonic")
        if (config.harmonic_count * 3000 * config.frequency_scale + 80 + 2 / config.call_duration
                >= config.sample_rate / 2):
            raise ValueError("The acoustic prior domain's highest harmonic exceeds Nyquist; reduce frequency_scale")
        super().__init__(config)
        with torch.no_grad():
            bias = self.network[-1].bias.view(config.splauts, 5)
            bias[:, 1] = 0  # Start all frequencies around the acoustic anchor.
            bias[:, 2] = -1.5
            bias[:, 3] = -1.5
            if config.harmonic_count > 1:
                # Interleave harmonic orders so every band spans the whole call.
                for h in range(config.harmonic_count):
                    count = len(bias[h::config.harmonic_count])
                    bias[h::config.harmonic_count, 0] = torch.linspace(-2, 2, count)
                    bias[h::config.harmonic_count, 4] -= 2 * config.harmonic_rolloff * torch.log(torch.tensor(h + 1.))

    def forward(self, codes: Tensor, durations: Tensor) -> Splauts:
        # Do not let padding or the frequency coordinate become unconstrained
        # shape inputs during inversion. Family and other coordinates are bounded.
        start = len(self.config.call_families)
        shape_codes = torch.cat((codes[:, :start], torch.zeros_like(codes[:, start:start+1]),
                                  codes[:, start+1:start+4], torch.zeros_like(codes[:, start+4:])), dim=-1)
        raw = self.network(shape_codes).reshape(-1, self.config.splauts, 5)
        values = acoustic_values(codes, self.config)
        centre, bandwidth = values[:, 0:1], values[:, 1:2]
        if "moo" in self.config.call_families:
            centre = centre - 0.5 * bandwidth * codes[:, self.config.call_families.index("moo") : self.config.call_families.index("moo")+1]
        orders = (torch.arange(self.config.splauts, device=codes.device) % self.config.harmonic_count + 1).to(codes.dtype)
        duration = durations[:, None]
        minimum_time = self.config.hop_length / self.config.sample_rate / 2
        minimum_frequency = self.config.sample_rate / self.config.n_fft / 2
        return Splauts(raw[..., 0].sigmoid() * duration,
                       orders * (centre + 0.5 * bandwidth * raw[..., 1].tanh()),
                       minimum_time + raw[..., 2].sigmoid() * (duration / 3 - minimum_time),
                       minimum_frequency + raw[..., 3].sigmoid() * (100 - minimum_frequency),
                       F.softplus(raw[..., 4]))
