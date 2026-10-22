"""Track-indexed field audio ingestion; optional scipy/soundfile dependencies."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf


@dataclass
class StationTable:
    ids: list[str]
    tracks: np.ndarray  # zero-based WAV channels; CSV TrackIndex is one-based
    xy: np.ndarray  # east/north metres relative to the anchor
    latlon: np.ndarray
    origin: np.ndarray


def read_stations(path: Path, anchor: str = "M04", count: int = 8) -> StationTable:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not 6 <= count <= 8:
        raise ValueError("Use 6–8 nearby stations")
    anchors = [r for r in rows if r["StationID"] == anchor]
    if len(anchors) != 1:
        raise ValueError("Anchor must identify exactly one metadata row")
    origin = np.array([float(anchors[0]["Lat"]), float(anchors[0]["Lon"])])
    latlon = np.array([[float(r["Lat"]), float(r["Lon"])] for r in rows])
    tracks = np.array([int(r["TrackIndex"]) - 1 for r in rows])
    if (len(np.unique(tracks)) != len(tracks) or (tracks < 0).any()
            or not np.isfinite(latlon).all()):
        raise ValueError("Metadata requires finite coordinates and unique positive track indices")
    xy = (latlon[:, ::-1] - origin[::-1]) * (np.pi / 180 * 6371000)
    xy[:, 0] *= np.cos(np.deg2rad(origin[0]))
    order = np.argsort(np.linalg.norm(xy, axis=1), kind="stable")[:count]
    if len(order) != count or np.linalg.matrix_rank(xy[order] - xy[order].mean(0)) < 2:
        raise ValueError("Insufficient stations or degenerate geometry")
    return StationTable([rows[i]["StationID"] for i in order], tracks[order],
                        xy[order], latlon[order], origin)


def load_audio(path: Path, stations: StationTable, sample_rate: int = 512,
               duration: float | None = None) -> tuple[np.ndarray, dict]:
    """Read bounded blocks with filter context before anti-aliased downsampling.

    All original channels must be decoded, but only selected channels are kept.
    Rational resampling and whole-second boundaries keep exact sample alignment.
    """
    with sf.SoundFile(path) as handle:
        if sample_rate < 256 or sample_rate > handle.samplerate:
            raise ValueError("Analysis rate must be >=256 Hz and <= the WAV rate")
        if stations.tracks.max() >= handle.channels:
            raise ValueError("CSV track index exceeds WAV channel count")
        total = len(handle) if duration is None else min(len(handle), round(duration * handle.samplerate))
        if total <= 0:
            raise ValueError("Audio duration must be positive")
        divisor = math.gcd(handle.samplerate, sample_rate)
        up, down = sample_rate // divisor, handle.samplerate // divisor
        block, context = handle.samplerate * 8, handle.samplerate
        chunks = []
        for start in range(0, total, block):
            end = min(start + block, total)
            left, right = max(0, start - context), min(len(handle), end + context)
            handle.seek(left)
            x = handle.read(right - left, dtype="float32", always_2d=True)[:, stations.tracks]
            if not np.isfinite(x).all():
                raise ValueError("Non-finite audio samples")
            y = signal.resample_poly(x, up, down, axis=0)
            offset = round((start - left) * sample_rate / handle.samplerate)
            length = math.ceil((end-start) * sample_rate / handle.samplerate)
            chunks.append(y[offset:offset + length])
            print(f"  decoded {end / handle.samplerate:.1f}s", flush=True)
        info = {"sample_rate": handle.samplerate, "channels": handle.channels,
                "duration_s": len(handle) / handle.samplerate, "subtype": handle.subtype,
                "analysis_rate": sample_rate, "analyzed_duration_s": total / handle.samplerate}
    return np.concatenate(chunks).T, info


def stft_power(audio: np.ndarray, sample_rate: int = 512, n_fft: int = 256,
               hop: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, z = signal.stft(audio, sample_rate, nperseg=n_fft,
                                      noverlap=n_fft - hop, boundary="zeros", padded=True)
    return z.transpose(0, 2, 1), times, frequencies


def audio_from_mask(z: np.ndarray, mask: np.ndarray, sample_rate: int = 512,
                    n_fft: int = 256, hop: int = 32) -> np.ndarray:
    """Reference-channel mixture phase; this does not recover clean source phase."""
    _, audio = signal.istft((z * mask).T, sample_rate, nperseg=n_fft,
                            noverlap=n_fft - hop, boundary=True)
    return audio


def window_starts(duration: float, window: float, hop: float) -> np.ndarray:
    if not 0 < hop <= window <= duration:
        raise ValueError("Require 0 < hop <= window <= audio duration")
    starts = np.arange(0, duration - window + 1e-8, hop)
    if duration - window - starts[-1] > 1e-6:
        starts = np.r_[starts, duration - window]
    return starts
