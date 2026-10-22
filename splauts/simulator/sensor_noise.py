"""Independent waveform sensor noise, including call-like local disturbances."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import Config, station_layout
from .calls import CallSpec, power_spectrogram, synthesize_call
from .scene import SourceEvent, simulate_scene


@dataclass
class SensorNoiseScene:
    stations: Tensor
    times: Tensor
    frequencies: Tensor
    observed: Tensor
    clean_shared: Tensor
    true_self: Tensor
    received_sources: Tensor
    source_power: Tensor
    source_times: Tensor
    events: list[SourceEvent]
    signal_waveforms: Tensor
    noise_waveforms: Tensor
    noise_gains: Tensor
    mode: str


def independent_sensor_noise(reference: Tensor, config: Config, seed: int) -> tuple[Tensor, Tensor]:
    """Per-channel independent colored floor, tonal burst and short disturbance.

    Local bursts are injected directly into each sensor waveform, without any
    travel delay, spatial rendering, or copied signal from another sensor.
    """
    times = torch.arange(reference.shape[1]) / config.sample_rate
    frequencies = torch.fft.rfftfreq(reference.shape[1], 1/config.sample_rate)
    response = .15 + torch.exp(-.5*((frequencies-45)/35).square())
    reference_rms = reference.square().mean().sqrt()
    waves, gains = [], []
    for m in range(len(reference)):
        rng = torch.Generator().manual_seed(seed+104729*m)
        white = torch.randn(reference.shape[1], generator=rng)
        colored = torch.fft.irfft(torch.fft.rfft(white)*response, n=len(white))
        phase = 6.28*float(torch.rand((), generator=rng))
        floor = colored*(.7+.3*torch.sin(2*torch.pi*times/config.scene_duration+phase))
        frequency = 22+24*float(torch.rand((), generator=rng))
        call = synthesize_call(CallSpec("moo", config.call_duration, frequency, 4.,
                                       harmonic_count=2), config.sample_rate)
        onset = .2+(config.scene_duration-config.call_duration-.4)*float(torch.rand((), generator=rng))
        start = round(onset*config.sample_rate)
        local = torch.zeros_like(white)
        local[start:start+len(call)] += call[:len(local)-start]
        impulse_time = .2+(config.scene_duration-.4)*float(torch.rand((), generator=rng))
        transient = torch.randn(len(times), generator=rng)*torch.exp(-.5*((times-impulse_time)/.1).square())
        wave = floor + 2.5*local + 2*transient
        wave = wave/wave.square().mean().sqrt().clamp_min(1e-12)
        gain = torch.exp(.65*torch.randn((), generator=rng)).clamp(.35, 3.)
        waves.append(wave*reference_rms*gain)
        gains.append(gain)
    return torch.stack(waves), torch.stack(gains)


def sensor_noise_scene(config: Config, seed=7, level=1., mode="additive_power",
                        noise_only=False, stations=8) -> SensorNoiseScene:
    if mode not in ("additive_power", "waveform") or level < 0:
        raise ValueError("Use nonnegative noise and additive_power or waveform mode")
    rng = torch.Generator().manual_seed(seed)
    fit_stations = station_layout(stations, "irregular", config.area_size)
    # Independent interior sensors are reserved for evaluation by the runner.
    held = config.area_size*torch.tensor([[.25,.25],[.75,.25],[.25,.75],[.75,.75]])
    all_stations = torch.cat((fit_stations, held))
    positions = config.area_size*torch.tensor([[.30,.37],[.71,.65]])
    positions += config.area_size*.07*(torch.rand((2,2), generator=rng)-.5)
    events = [SourceEvent(positions[0], .6, CallSpec("moo",config.call_duration,29.,5.,harmonic_count=3)),
              SourceEvent(positions[1], 1.25, CallSpec("moo",config.call_duration,38.,7.,harmonic_count=3), .85)]
    scenes = [simulate_scene([event], all_stations, config) for event in events]
    signal = sum(scene.waveforms for scene in scenes)
    noise, gains = independent_sensor_noise(signal, config, seed+10000)
    noise = level*noise
    self_power, times, frequencies = power_spectrogram(noise, config)
    received = torch.stack([scene.power for scene in scenes])
    clean = received.sum(0)
    if noise_only:
        signal = torch.zeros_like(signal)
        clean = torch.zeros_like(clean)
        received = torch.zeros_like(received)
        events = []
    observed = clean+self_power if mode=="additive_power" else power_spectrogram(signal+noise,config)[0]
    source_powers = torch.stack([power_spectrogram(s.source_waveforms[0],config)[0] for s in scenes])
    source_times = scenes[0].source_times[0]
    band = (frequencies>=8)&(frequencies<=120)
    return SensorNoiseScene(all_stations,times,frequencies[band],observed[...,band],clean[...,band],
                            self_power[...,band],received[...,band],source_powers[...,band],source_times,
                            events,signal,noise,gains,mode)
