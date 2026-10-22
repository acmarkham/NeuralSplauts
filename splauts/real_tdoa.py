"""Independent band-limited waveform delay-and-sum correlation localization.

Uses the same six fitting stations, never the two validation channels. A broad
or multimodal map is not a calibrated location posterior. No source identities
or clean audio are inferred by this baseline.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import torch


def spatial_correlation(audio, stations, candidates, rate=512, speed=343.):
    """Normalized raw cross-correlation; fractional travel delays, no PHAT claim."""
    values = np.zeros(len(candidates))
    diagnostics = []
    distances = np.linalg.norm(candidates[:,None]-stations[None], axis=-1)
    pairs = list(itertools.combinations(range(len(stations)), 2))
    for i, j in pairs:
        a, b = audio[i]-audio[i].mean(), audio[j]-audio[j].mean()
        c = signal.correlate(a, b, method="fft")
        c /= max(np.linalg.norm(a)*np.linalg.norm(b), 1e-20)
        lags = signal.correlation_lags(len(a), len(b))/rate
        values += np.interp((distances[:,i]-distances[:,j])/speed, lags, c)
        bound = np.linalg.norm(stations[i]-stations[j])/speed
        valid = np.abs(lags) <= bound
        indices = np.flatnonzero(valid)
        peak = indices[np.argmax(c[valid])]
        diagnostics.append({"i": i, "j": j, "physical_limit_s": float(bound),
                            "peak_physical_lag_s": float(lags[peak]), "peak_correlation": float(c[peak]),
                            "unconstrained_peak_lag_s": float(lags[np.argmax(c)])})
    return values/len(pairs), diagnostics


def run(directory):
    protocol = json.loads((directory/"protocol.json").read_text())
    results = json.loads((directory/"results.json").read_text())
    with np.load(directory/"audio_cache.npz") as data:
        audio = data["audio"]
    audio = signal.sosfiltfilt(signal.butter(4, [20,110], btype="bandpass", fs=512, output="sos"), audio, axis=1)
    labels = [s["id"] for s in protocol["stations"]]
    train = np.array([labels.index(label) for label in protocol["fit_stations"]])
    xy = np.array([s["east_north_m"] for s in protocol["stations"]])
    area = protocol["config"]["area_size"]
    origin = (xy.min(0)+xy.max(0)-area)/2
    east = np.linspace(origin[0], origin[0]+area, 101)
    north = np.linspace(origin[1], origin[1]+area, 101)
    yy, xx = np.meshgrid(north, east, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    audit = []
    for start in sorted({r["window_start_s"] for r in results}):
        first, last = round(start*512), round((start+protocol["arguments"]["window"])*512)
        scores, diagnostics = spatial_correlation(audio[train, first:last], xy[train], grid)
        best = int(scores.argmax())
        row = {"window_start_s": start, "peak_east_north_m": grid[best].tolist(),
               "peak_mean_pair_correlation": float(scores[best]),
               "median_mean_pair_correlation": float(np.median(scores)), "pairs": diagnostics}
        audit.append(row)
        fig, ax = plt.subplots(figsize=(7,7), constrained_layout=True)
        im = ax.imshow(scores.reshape(101,101), origin="lower", extent=[east[0],east[-1],north[0],north[-1]], cmap="viridis")
        ax.scatter(*xy[train].T, marker="^", color="white")
        ax.scatter(*grid[best], marker="x", color="red", s=100, label="waveform correlation maximum")
        matches = [r for r in results if r["window_start_s"] == start]
        for r in matches:
            saved = torch.load(directory/r["directory"]/"fit.pt", weights_only=True)
            points = saved["state"]["positions"].numpy()+origin
            ax.scatter(*points.T, marker="*", s=100, label=f"splauts K={r['K']}")
        for i in train:
            ax.annotate(labels[i], xy[i], color="white")
        ax.set(xlabel="East of M04 (m)", ylabel="North of M04 (m)", title=f"{start:g} s: 20–110 Hz waveform correlation")
        ax.legend(fontsize=8)
        fig.colorbar(im, ax=ax, label="Mean normalized pair correlation")
        fig.savefig(directory/f"tdoa_{start:06.2f}.png", dpi=120)
        plt.close(fig)
    (directory/"tdoa.json").write_text(json.dumps({"band_hz":[20,110], "grid_spacing_m":area/100,
        "method":"Mean normalized raw waveform pair correlation at predicted TDOAs; fit stations only",
        "caveat":"One strongest peak per window, no animal identification, and no calibrated uncertainty or source-separation claim",
        "windows":audit}, indent=2), encoding="utf-8")
    print(f"Saved {len(audit)} independent waveform-correlation maps", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/real_rumbles"))
    run(parser.parse_args().output)
