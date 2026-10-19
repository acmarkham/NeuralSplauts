"""Waveform simulation, deliberately independent of the additive power renderer."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..config import Config, station_layout, validate_stations
from .calls import CallSpec, power_spectrogram, sample_call, synthesize_call


@dataclass
class SourceEvent:
    position: Tensor
    onset: float
    call: CallSpec
    amplitude: float = 1.0  # Waveform amplitude; renderer uses its square.


@dataclass
class Scene:
    stations: Tensor
    events: list[SourceEvent]
    waveforms: Tensor
    power: Tensor
    times: Tensor
    frequencies: Tensor
    source_waveforms: list[Tensor]
    source_power: list[Tensor]
    source_times: list[Tensor]
    delays: Tensor  # [stations, events], excluding emission onset
    power_gains: Tensor


def fractional_delay(waveform: Tensor, delay_seconds: float | Tensor,
                     output_samples: int, sample_rate: int) -> Tensor:
    """Bandlimited fractional delay via a phase ramp on a zero-padded waveform.

    Padding keeps the shifted waveform away from the circular boundary. Sinc
    interpolation has tiny tails; unlike linear interpolation it preserves
    call amplitude at high frequencies and arbitrary fractional-sample offsets.
    """
    if waveform.ndim != 1 or not len(waveform) or output_samples < 1 or sample_rate < 1:
        raise ValueError("Supply a nonempty waveform and positive output length/sample rate")
    shift = torch.as_tensor(delay_seconds, dtype=waveform.dtype) * sample_rate
    if not torch.isfinite(shift) or shift < 0:
        raise ValueError("Delay must be finite and nonnegative")
    size = output_samples + len(waveform) + math.ceil(float(shift))
    fft_size = 1 << (size - 1).bit_length()
    frequencies = torch.fft.rfftfreq(fft_size).to(waveform.dtype)
    phase = torch.exp(-2j * torch.pi * frequencies * shift)
    return torch.fft.irfft(torch.fft.rfft(waveform, n=fft_size) * phase, n=fft_size)[:output_samples]


def simulate_scene(events: list[SourceEvent], stations: Tensor, config: Config,
                   seed: int | None = None) -> Scene:
    validate_stations(stations)
    if not events:
        raise ValueError("At least one event is required")
    positions = torch.stack([torch.as_tensor(e.position, dtype=stations.dtype) for e in events])
    if positions.shape != (len(events), 2) or not torch.isfinite(positions).all():
        raise ValueError("Event positions must be finite 2D coordinates")
    if ((positions < 0) | (positions > config.area_size)).any():
        raise ValueError("Events must lie within the observation area")
    distances = torch.cdist(stations, positions)
    delays = distances / config.sound_speed
    gains = (distances + config.distance_offset).pow(-2)
    count = round(config.scene_duration * config.sample_rate)
    waveforms = torch.zeros(len(stations), count, dtype=stations.dtype)
    source_waves, source_powers, source_times = [], [], []
    for k, event in enumerate(events):
        if event.onset < 0 or event.amplitude <= 0:
            raise ValueError("Onsets must be nonnegative and amplitudes positive")
        if event.onset + float(delays[:, k].max()) + event.call.duration > config.scene_duration:
            raise ValueError("Scene is too short to preserve all arrivals")
        source = synthesize_call(event.call, config.sample_rate).to(stations.dtype)
        power, times, _ = power_spectrogram(source, config)
        source_waves.append(source)
        source_powers.append(power)
        source_times.append(times)
        for m in range(len(stations)):
            delayed = fractional_delay(source, event.onset + delays[m, k], count, config.sample_rate)
            waveforms[m] += event.amplitude * gains[m, k].sqrt() * delayed
    rng = torch.Generator().manual_seed(config.seed if seed is None else seed)
    waveforms += config.noise_std * torch.randn(waveforms.shape, generator=rng, dtype=waveforms.dtype)
    power, times, frequencies = power_spectrogram(waveforms, config)
    return Scene(stations, events, waveforms, power, times, frequencies,
                 source_waves, source_powers, source_times, delays, gains)


def random_scene(config: Config, event_count: int = 1, station_count: int = 4,
                 layout: str = "corners", call_type: str | None = None) -> Scene:
    if event_count not in (1, 2, 3):
        raise ValueError("The prototype supports 1–3 calls per random scene")
    rng = torch.Generator().manual_seed(config.seed)
    events = []
    for k in range(event_count):
        kind = call_type or ("tonal", "sweep", "pulse")[k % 3]
        position = config.area_size * (0.25 + 0.5 * torch.rand(2, generator=rng))
        spec = sample_call(kind, config, rng)
        events.append(SourceEvent(position, 0.2 + 0.13 * k, spec,
                                  0.8 + 0.4 * float(torch.rand((), generator=rng))))
    return simulate_scene(events, station_layout(station_count, layout, config.area_size), config)
