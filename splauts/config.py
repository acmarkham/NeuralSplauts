"""Physical conventions and reproducible station geometries (metres, seconds, Hz)."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class Config:
    sample_rate: int = 8000
    n_fft: int = 512
    hop_length: int = 80
    scene_duration: float = 2.4
    call_duration: float = 0.4
    area_size: float = 300.0
    sound_speed: float = 343.0
    distance_offset: float = 1.0
    noise_std: float = 0.0  # Absolute waveform amplitude, after attenuation.
    splauts: int = 32
    latent_dim: int = 16
    seed: int = 7

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.n_fft < 4 or self.hop_length <= 0:
            raise ValueError("Invalid STFT configuration")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must not exceed n_fft")
        if min(self.scene_duration, self.call_duration, self.area_size,
               self.sound_speed, self.distance_offset) <= 0:
            raise ValueError("Durations, area, sound speed and distance offset must be positive")
        if self.noise_std < 0 or self.splauts < 1 or self.latent_dim < 1:
            raise ValueError("Noise must be nonnegative and model dimensions positive")


def validate_stations(stations: Tensor) -> None:
    """Primary experiments require >=4 finite, distinct, non-collinear stations."""
    if stations.ndim != 2 or stations.shape[1] != 2 or len(stations) < 4:
        raise ValueError("Use at least four stations with shape [stations, 2]")
    if not torch.isfinite(stations).all():
        raise ValueError("Station coordinates must be finite")
    if len(torch.unique(stations, dim=0)) != len(stations):
        raise ValueError("Stations must have distinct coordinates")
    if torch.linalg.matrix_rank(stations - stations.mean(0)) < 2:
        raise ValueError("Stations must not be collinear")


def station_layout(count: int = 4, kind: str = "corners", area_size: float = 300.0) -> Tensor:
    """Perimeter layouts, plus a deliberately clustered geometry for stress tests."""
    if count < 4 or area_size <= 0:
        raise ValueError("Use at least four stations and a positive area size")
    if kind == "corners":
        # Walk the square perimeter; 4, 6 and 8 all span the observation area.
        s = torch.arange(count, dtype=torch.float32) * 4 / count
        x = torch.where(s < 1, s, torch.where(s < 2, 1., torch.where(s < 3, 3-s, 0.)))
        y = torch.where(s < 1, 0., torch.where(s < 2, s-1, torch.where(s < 3, 1., 4-s)))
        stations = torch.stack((x, y), -1) * area_size
    elif kind in ("irregular", "clustered"):
        angles = torch.arange(count) * (2 * math.pi / count) + 0.17
        radii = 0.43 + 0.04 * torch.sin(3 * angles + 0.4)
        stations = area_size * (0.5 + radii[:, None] * torch.stack((angles.cos(), angles.sin()), -1))
        if kind == "clustered":
            stations = stations * 0.12 + area_size * 0.05
    else:
        raise ValueError(f"Unknown station layout: {kind}")
    validate_stations(stations)
    return stations
