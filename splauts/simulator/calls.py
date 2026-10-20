"""Simple procedural calls inspired by the call families in the design plan.

These are self-contained illustrative oscillators, not a reproduction of an
external Spore/Skog implementation or a model of a particular species.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..config import Config

CALL_TYPES = ("tonal", "sweep", "pulse", "moo")


@dataclass(frozen=True)
class CallSpec:
    kind: str = "tonal"
    duration: float = 0.4
    frequency: float = 1400.0
    bandwidth: float = 240.0
    modulation_hz: float = 5.0
    pulse_hz: float = 16.0
    harmonic_count: int = 1
    harmonic_rolloff: float = 0.7


def instantaneous_frequency(spec: CallSpec, times: Tensor) -> Tensor:
    """Fundamental contour in Hz; moo bandwidth is the main-to-edge pitch drop."""
    if spec.kind == "moo":
        u = (times / spec.duration).clamp(0, 1)
        rise = 0.5 * (1 - torch.cos(math.pi * (u / 0.25).clamp(0, 1)))
        fall = 0.5 * (1 - torch.cos(math.pi * ((1 - u) / 0.25).clamp(0, 1)))
        return spec.frequency - spec.bandwidth * (1 - rise * fall)
    if spec.kind == "sweep":
        return spec.frequency + spec.bandwidth * (times / spec.duration - 0.5)
    if spec.kind == "tonal":
        return spec.frequency + 0.12 * spec.bandwidth * torch.cos(2 * math.pi * max(spec.modulation_hz, 0.1) * times)
    return torch.full_like(times, spec.frequency)


def moo_phase(spec: CallSpec, times: Tensor) -> Tensor:
    """Analytic integral of a cosine rise, plateau and cosine fall; continuous phase."""
    edge = spec.duration * 0.25
    f, b = spec.frequency, spec.bandwidth
    rise = (f - b / 2) * times - b * edge / (2 * math.pi) * torch.sin(math.pi * times / edge)
    middle = (f - b / 2) * edge + f * (times - edge)
    v = times - (spec.duration - edge)
    fall = ((f - b / 2) * edge + f * (spec.duration - 2 * edge)
            + (f - b / 2) * v + b * edge / (2 * math.pi) * torch.sin(math.pi * v / edge))
    return 2 * math.pi * torch.where(times < edge, rise,
                                    torch.where(times > spec.duration - edge, fall, middle))


def synthesize_call(spec: CallSpec, sample_rate: int = 8000) -> Tensor:
    if spec.kind not in CALL_TYPES:
        raise ValueError(f"Unknown call type: {spec.kind}")
    if spec.duration <= 0 or sample_rate <= 0 or spec.pulse_hz <= 0:
        raise ValueError("Duration, sample rate and pulse rate must be positive")
    if (not isinstance(spec.harmonic_count, int) or spec.harmonic_count < 1
            or not math.isfinite(spec.harmonic_rolloff) or spec.harmonic_rolloff < 0):
        raise ValueError("Use a positive integer harmonic count and nonnegative rolloff")
    if spec.frequency - abs(spec.bandwidth) <= 0 or spec.frequency + abs(spec.bandwidth) >= sample_rate / 2:
        raise ValueError("Call frequencies must lie inside the Nyquist band")
    # Reserve room for pulse AM sidebands and the smooth finite envelope.
    # FM has infinite weak sidebands; the conservative bandwidth guard keeps
    # the significant energy away from Nyquist rather than clipping harmonics.
    guard = (4 * spec.pulse_hz if spec.kind == "pulse" else 0) + 2 / spec.duration
    if (spec.harmonic_count > 1 and
            spec.harmonic_count * (spec.frequency + abs(spec.bandwidth)) + guard >= sample_rate / 2):
        raise ValueError("Highest harmonic and modulation sidebands must lie below Nyquist")
    t = torch.arange(round(spec.duration * sample_rate)) / sample_rate
    u = t / spec.duration
    envelope = torch.sin(math.pi * u).square()
    if spec.kind == "moo":
        phase = moo_phase(spec, t)
        # A sustained body makes the pitch at the attack/release audible.
        envelope = (torch.sin(math.pi / 2 * (u / 0.15).clamp(0, 1)).square()
                    * torch.sin(math.pi / 2 * ((1-u) / 0.15).clamp(0, 1)).square())
    elif spec.kind == "sweep":
        phase = 2 * math.pi * ((spec.frequency - spec.bandwidth / 2) * t
                              + spec.bandwidth * t.square() / (2 * spec.duration))
    elif spec.kind == "tonal":
        rate = max(spec.modulation_hz, 0.1)
        phase = 2 * math.pi * spec.frequency * t + 0.12 * spec.bandwidth / rate * torch.sin(2 * math.pi * rate * t)
    else:
        phase = 2 * math.pi * spec.frequency * t
        envelope = envelope * (0.5 + 0.5 * torch.cos(2 * math.pi * spec.pulse_hz * t)).pow(4)
    orders = torch.arange(1, spec.harmonic_count + 1, dtype=t.dtype)
    amplitudes = orders.pow(-spec.harmonic_rolloff)
    carrier = (amplitudes[:, None] * torch.sin(orders[:, None] * phase)).sum(0)
    return envelope * carrier / amplitudes.square().sum().sqrt()


def sample_call(kind: str, config: Config, rng: torch.Generator) -> CallSpec:
    values = torch.rand(3, generator=rng)
    return CallSpec(kind, config.call_duration, config.frequency_scale * (900 + 1400 * float(values[0])),
                    config.frequency_scale * (140 + 280 * float(values[1])), 3 + 4 * float(values[2]),
                    harmonic_count=config.harmonic_count, harmonic_rolloff=config.harmonic_rolloff)


def power_spectrogram(waveform: Tensor, config: Config) -> tuple[Tensor, Tensor, Tensor]:
    """Return [..., time, frequency] power and shared axes; centred, zero-padded STFT."""
    window = torch.hann_window(config.n_fft, dtype=waveform.dtype, device=waveform.device)
    stft = torch.stft(waveform, config.n_fft, config.hop_length, window=window,
                      center=True, pad_mode="constant", normalized=True, return_complex=True)
    power = stft.abs().square().transpose(-1, -2)
    times = torch.arange(power.shape[-2], dtype=waveform.dtype, device=waveform.device) * config.hop_length / config.sample_rate
    frequencies = torch.fft.rfftfreq(config.n_fft, 1 / config.sample_rate, device=waveform.device).to(waveform.dtype)
    return power, times, frequencies
