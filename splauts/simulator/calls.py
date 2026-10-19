"""Simple procedural calls inspired by the call families in the design plan.

These are self-contained illustrative oscillators, not a reproduction of an
external Spore/Skog implementation or a model of a particular species.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..config import Config

CALL_TYPES = ("tonal", "sweep", "pulse")


@dataclass(frozen=True)
class CallSpec:
    kind: str = "tonal"
    duration: float = 0.4
    frequency: float = 1400.0
    bandwidth: float = 240.0
    modulation_hz: float = 5.0
    pulse_hz: float = 16.0


def synthesize_call(spec: CallSpec, sample_rate: int = 8000) -> Tensor:
    if spec.kind not in CALL_TYPES:
        raise ValueError(f"Unknown call type: {spec.kind}")
    if spec.duration <= 0 or sample_rate <= 0 or spec.pulse_hz <= 0:
        raise ValueError("Duration, sample rate and pulse rate must be positive")
    if spec.frequency - abs(spec.bandwidth) <= 0 or spec.frequency + abs(spec.bandwidth) >= sample_rate / 2:
        raise ValueError("Call frequencies must lie inside the Nyquist band")
    t = torch.arange(round(spec.duration * sample_rate)) / sample_rate
    u = t / spec.duration
    envelope = torch.sin(math.pi * u).square()
    if spec.kind == "sweep":
        phase = 2 * math.pi * ((spec.frequency - spec.bandwidth / 2) * t
                              + spec.bandwidth * t.square() / (2 * spec.duration))
    elif spec.kind == "tonal":
        rate = max(spec.modulation_hz, 0.1)
        phase = 2 * math.pi * spec.frequency * t + 0.12 * spec.bandwidth / rate * torch.sin(2 * math.pi * rate * t)
    else:
        phase = 2 * math.pi * spec.frequency * t
        envelope = envelope * (0.5 + 0.5 * torch.cos(2 * math.pi * spec.pulse_hz * t)).pow(4)
    return envelope * torch.sin(phase)


def sample_call(kind: str, config: Config, rng: torch.Generator) -> CallSpec:
    values = torch.rand(3, generator=rng)
    return CallSpec(kind, config.call_duration, 900 + 1400 * float(values[0]),
                    140 + 280 * float(values[1]), 3 + 4 * float(values[2]))


def power_spectrogram(waveform: Tensor, config: Config) -> tuple[Tensor, Tensor, Tensor]:
    """Return [..., time, frequency] power and shared axes; centred, zero-padded STFT."""
    window = torch.hann_window(config.n_fft, dtype=waveform.dtype, device=waveform.device)
    stft = torch.stft(waveform, config.n_fft, config.hop_length, window=window,
                      center=True, pad_mode="constant", normalized=True, return_complex=True)
    power = stft.abs().square().transpose(-1, -2)
    times = torch.arange(power.shape[-2], dtype=waveform.dtype, device=waveform.device) * config.hop_length / config.sample_rate
    frequencies = torch.fft.rfftfreq(config.n_fft, 1 / config.sample_rate, device=waveform.device).to(waveform.dtype)
    return power, times, frequencies
