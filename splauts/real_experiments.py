"""Windowed, mixture-only low-frequency field-data feasibility experiment.

Run ``python -m splauts.real_experiments --help``. No elephant labels or clean
calls are available: inferred components are candidates, not identified animals.
"""

import argparse
import csv
from dataclasses import asdict
import html
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import soundfile as sf
import torch

from .config import Config
from .model.generator import Splauts, evaluate_field
from .model.renderer import render
from .real_data import (read_stations, load_audio, stft_power, audio_from_mask,
                        window_starts)
from .training.mixtures import MixtureObservations, fit_mixtures, mixture_losses


def score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(mixture_losses(prediction, target)[0])


def render_components(fit, stations, times, frequencies, config):
    with torch.no_grad():
        return torch.stack([
            render(Splauts(**{key: getattr(fit.splauts, key)[k:k+1]
                              for key in asdict(fit.splauts)}),
                   fit.state.positions[k:k+1], fit.state.onsets[k:k+1],
                   fit.state.amplitudes[k:k+1], stations, times, frequencies, config)
            for k in range(len(fit.state.positions))])


def make_overview(output, stations, audio, z, times, frequencies, band, background, train):
    fig, axes = plt.subplots(len(stations.ids) + 1, 1, figsize=(13, 17), constrained_layout=True)
    ax = axes[0]
    ax.scatter(*stations.xy[train].T, marker="^", label="fit")
    held = np.setdiff1d(np.arange(len(stations.ids)), train)
    ax.scatter(*stations.xy[held].T, marker="s", label="held out")
    for label, xy in zip(stations.ids, stations.xy):
        ax.annotate(label, xy)
    ax.set(xlabel="East of M04 (m)", ylabel="North of M04 (m)", aspect="equal")
    ax.legend()
    power = np.abs(z[..., band]) ** 2
    vmax = np.quantile(power, .999)
    for m, ax in enumerate(axes[1:]):
        ax.pcolormesh(times, frequencies[band], 10*np.log10(np.maximum(power[m].T, vmax*1e-7)/vmax),
                      shading="auto", vmin=-45, vmax=0)
        ax.set(ylabel=f"{stations.ids[m]} Hz", ylim=(frequencies[band][0], frequencies[band][-1]))
    axes[-1].set_xlabel("Seconds from WAV start")
    fig.savefig(output / "overview.png", dpi=120)
    plt.close(fig)


def save_case(directory, fit, observation, components, z, frequencies, band,
              background, config, train, held, offset, window, stations, origin):
    directory.mkdir(parents=True, exist_ok=True)
    prediction = components.sum(0)
    target = torch.as_tensor(observation)
    residual = target - prediction
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    extent = [offset, offset + window, frequencies[band][0], frequencies[band][-1]]
    # Render entire padded support; x limits identify the requested arrival-time window.
    t = np.arange(observation.shape[1]) * config.hop_length/config.sample_rate + origin
    for ax, value, title in zip(axes, (target[0], prediction[0], residual[0]),
                                ("Observed excess power at M04", "Sum of inferred source powers", "Signed residual")):
        scale = float(target[0].max())
        ax.pcolormesh(t, frequencies[band], value.T.numpy()/max(scale, 1e-12), shading="auto",
                      cmap="coolwarm" if title == "Signed residual" else "magma")
        ax.set(title=title, xlim=extent[:2], ylabel="Hz")
    axes[-1].set_xlabel("Seconds from WAV start")
    fig.savefig(directory / "reconstruction.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(*stations.xy[train].T, marker="^", label="fit stations")
    ax.scatter(*stations.xy[held].T, marker="s", label="held-out stations")
    for label, xy in zip(stations.ids, stations.xy):
        ax.annotate(label, xy)
    xy = fit.state.positions.numpy() + offset_xy(config, stations)
    ax.scatter(*xy.T, marker="*", s=150, label="inferred candidates")
    for k, p in enumerate(xy):
        ax.annotate(str(k + 1), p)
    ax.set(xlabel="East of M04 (m)", ylabel="North of M04 (m)", aspect="equal")
    ax.legend()
    fig.savefig(directory / "map.png", dpi=120)
    plt.close(fig)
    # Mixture-phase masks are reference-channel estimates; residual is retained.
    raw_power = np.abs(z[0]) ** 2
    denom = np.maximum(raw_power[:, band], prediction[0].numpy() + background[0])
    masks = components[:, 0].numpy() / np.maximum(denom, 1e-20)[None]
    ref_mask = np.zeros_like(raw_power)
    ref_mask[:, band] = 1
    mixture = audio_from_mask(z[0], ref_mask, config.sample_rate, config.n_fft, config.hop_length)
    waves = []
    for mask in masks:
        full = np.zeros_like(raw_power)
        full[:, band] = mask
        waves.append(audio_from_mask(z[0], full, config.sample_rate, config.n_fft, config.hop_length))
    waves.append(mixture - np.sum(waves, axis=0))
    first = max(0, round((offset - origin) * config.sample_rate))
    last = first + round(window * config.sample_rate)
    # Same gain for mixture/components/residual; a 4x playback copy makes infrasound audible.
    gain = .8 / max(float(np.max(np.abs(mixture[first:last]))), 1e-12)
    for name, wave in [("mixture", mixture), *[(f"component_{k+1}", w) for k, w in enumerate(waves[:-1])],
                       ("residual", waves[-1])]:
        sf.write(directory / f"{name}.wav", wave[first:last] * gain, config.sample_rate, subtype="FLOAT")
        sf.write(directory / f"{name}_4x.wav", wave[first:last] * gain, config.sample_rate*4, subtype="FLOAT")
    torch.save({"config": asdict(config), "generator": fit.generator.state_dict(),
                "state": asdict(fit.state), "splauts": asdict(fit.splauts),
                "times": torch.arange(observation.shape[1]) * config.hop_length/config.sample_rate,
                "frequencies": torch.as_tensor(frequencies[band].copy()),
                "stations": torch.as_tensor(stations.xy - offset_xy(config, stations), dtype=torch.float32),
                "observation": target, "components": components, "train_indices": train.tolist(),
                "heldout_indices": held.tolist(), "audio_time_origin_s": origin,
                "coordinate_origin_east_north_m": offset_xy(config, stations).tolist(),
                "background": torch.as_tensor(background), "objectives": fit.objectives,
                "best_step": fit.best_step}, directory / "fit.pt")


def offset_xy(config, stations):
    return (stations.xy.min(0) + stations.xy.max(0) - config.area_size) / 2


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    stations = read_stations(args.metadata or args.audio.with_suffix(".csv"), args.anchor, args.stations)
    cache = output / "audio_cache.npz"
    signature = {"path": str(args.audio.resolve()), "size": args.audio.stat().st_size,
                 "mtime": args.audio.stat().st_mtime_ns, "tracks": stations.tracks.tolist(),
                 "duration": args.duration, "rate": 512}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if json.loads(str(data["signature"])) != signature:
                raise ValueError("Audio cache mismatch; use another output directory")
            audio, info = data["audio"], json.loads(str(data["info"]))
    else:
        audio, info = load_audio(args.audio, stations, duration=args.duration)
        np.savez_compressed(cache, audio=audio, signature=json.dumps(signature), info=json.dumps(info))
    z, times, frequencies = stft_power(audio)
    band = (frequencies >= args.low_hz) & (frequencies <= args.high_hz)
    if band.sum() < 3:
        raise ValueError("Band must contain at least three FFT bins")
    raw = np.abs(z[..., band]) ** 2
    # Fixed per-station stationary noise floor. All stations receive the same scalar scaling.
    background = np.quantile(raw, .2, axis=1, keepdims=True).astype(np.float32)
    scale = float(np.quantile(raw, .99))
    if scale <= 0:
        raise ValueError("Empty acoustic energy")
    power = np.maximum(raw - background, 0) / scale
    background /= scale
    z /= np.sqrt(scale)
    held = np.array([2, min(5, args.stations - 1)])
    train = np.setdiff1d(np.arange(args.stations), held)
    make_overview(output, stations, audio, z, times, frequencies, band, background, train)
    area = float(np.ptp(stations.xy, axis=0).max() + 2*args.margin)
    config = Config(sample_rate=512, n_fft=256, hop_length=32, splauts=args.splauts,
                    call_duration=args.call_duration, area_size=area, distance_offset=args.distance_offset)
    origin_xy = offset_xy(config, stations)
    positions = torch.as_tensor(stations.xy - origin_xy, dtype=torch.float32)
    # Pad for every source/station travel time anywhere inside the search square.
    corners = torch.tensor([[0., 0.], [0., area], [area, 0.], [area, area]])
    pad = np.ceil(float(torch.cdist(corners, positions).max()) / config.sound_speed + .5)
    starts = window_starts(audio.shape[1]/512, args.window, args.hop)
    if args.starts is not None:
        starts = np.array(args.starts)
        if (starts < 0).any() or (starts + args.window > audio.shape[1]/512 + 1e-6).any():
            raise ValueError("Requested windows fall outside analyzed audio")
    protocol = {"audio": info, "input": signature, "stations": [
        {"id": label, "track_index": int(track+1), "east_north_m": xy.tolist(), "lat_lon": ll.tolist()}
        for label, track, xy, ll in zip(stations.ids, stations.tracks, stations.xy, stations.latlon)],
        "fit_stations": [stations.ids[i] for i in train], "heldout_stations": [stations.ids[i] for i in held],
        "config": asdict(config), "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "padding_s": float(pad), "power_scale": scale,
        "caveats": ["No elephant identity, event count, source audio or position ground truth.",
                    "K is a supplied capacity prior, not a detected animal count.",
                    "20th percentile background uses the whole recording, including held-out station calibration.",
                    "Held-out stations never enter fitting, initialization or checkpoint selection.",
                    "Window overlaps duplicate candidates; catalog rows are not independent events.",
                    "Mixture phase masks at M04 do not establish clean source recovery.",
                    "Perfect synchronization, fixed sound speed, isotropic spreading and stationary sources are assumptions."]}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results, catalog = [], []
    for wi, start in enumerate(starts):
        first = max(0, int(np.floor((start-pad)*16)))
        last = min(len(times), int(np.ceil((start+args.window+pad)*16))+1)
        origin = float(times[first])
        local_times = torch.arange(last-first, dtype=torch.float32)/16
        local_power = torch.as_tensor(power[:, first:last].copy())
        obs = MixtureObservations(local_power[train], positions[train], local_times,
                                  torch.as_tensor(frequencies[band].copy(), dtype=torch.float32))
        for count in args.events:
            name = f"window_{start:06.2f}_K{count}"
            print(f"[{wi+1}/{len(starts)}] {name}", flush=True)
            tic = time.perf_counter()
            fit = fit_mixtures(obs, config, count, args.warmup_steps, args.steps,
                               geometry_steps=0, proposal_grid_size=args.grid_size,
                               fit_seed=args.seed, bandwidth_hz=args.high_hz-args.low_hz, log_every=0)
            components = render_components(fit, positions, local_times, obs.frequencies, config)
            prediction = components.sum(0)
            # Score only the core arrival-time interval, not easy padded silence.
            core = (local_times >= start-origin) & (local_times < start+args.window-origin)
            train_nmse = score(prediction[train][:, core], local_power[train][:, core])
            held_nmse = score(prediction[held][:, core], local_power[held][:, core])
            stationary = local_power.mean(1, keepdim=True).expand_as(local_power)
            row = {"window_start_s": float(start), "K": count, "directory": name,
                   "fit_nmse": train_nmse, "heldout_nmse": held_nmse,
                   "heldout_stationary_oracle_nmse": score(stationary[held][:,core], local_power[held][:,core]),
                   "best_step": fit.best_step, "seconds": time.perf_counter()-tic}
            for k in range(count):
                field = evaluate_field(Splauts(**{key: getattr(fit.splauts, key)[k:k+1]
                                                   for key in asdict(fit.splauts)}),
                                       fit.proposals.relative_times, obs.frequencies)[0]
                energy = field.sum(1)
                centroid = float((energy*fit.proposals.relative_times).sum()/energy.sum())
                xy = fit.state.positions[k].numpy()+origin_xy
                lat = stations.origin[0]+xy[1]/111194.9266
                lon = stations.origin[1]+xy[0]/(111194.9266*np.cos(np.deg2rad(stations.origin[0])))
                catalog.append({"window_start_s": float(start), "K": count, "component": k+1,
                                "east_m": float(xy[0]), "north_m": float(xy[1]), "lat": float(lat), "lon": float(lon),
                                "emission_centroid_s": origin+float(fit.state.onsets[k])+centroid,
                                "reference_core_power_fraction": float(components[k,0,core].sum()/local_power[0,core].sum().clamp_min(1e-12))})
            save_case(output/name, fit, local_power.numpy(), components, z[:, first:last], frequencies,
                      band, background, config, train, held, float(start), args.window, stations, origin)
            (output/name/"metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            results.append(row)
            print(f"  fit NMSE {train_nmse:.3f}; held-out {held_nmse:.3f}", flush=True)
            (output/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (output/"candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)
    make_report(output, protocol, results, catalog)
    return results


def make_report(output, protocol, results, catalog):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for count in sorted({r["K"] for r in results}):
        rows = [r for r in results if r["K"] == count]
        for ax, key in zip(axes, ("fit_nmse", "heldout_nmse")):
            ax.plot([r["window_start_s"] for r in rows], [r[key] for r in rows], ".-", label=f"K={count}")
            ax.set(xlabel="Window start (s)", ylabel="Station-balanced power NMSE", title=key)
            ax.axhline(1, color="gray", linestyle="--", label="zero excess prediction" if count == min(r["K"] for r in results) else None)
            ax.legend()
    fig.savefig(output/"summary.png", dpi=130)
    plt.close(fig)
    caveats = "".join(f"<li>{html.escape(s)}</li>" for s in protocol["caveats"])
    findings = "<h2>Measured reconstruction</h2><ul>"
    for count in sorted({r["K"] for r in results}):
        group = [r for r in results if r["K"] == count]
        findings += (f'<li>K={count}: median fit NMSE {np.median([r["fit_nmse"] for r in group]):.3f}; '
                     f'median held-out NMSE {np.median([r["heldout_nmse"] for r in group]):.3f}; '
                     f'{sum(r["heldout_nmse"] < 1 for r in group)}/{len(group)} below zero excess prediction.</li>')
    findings += "</ul><p>A lower reconstruction error does not identify separate elephants or establish location accuracy. In this first pass, prediction at held-out stations remains weak: these candidates should not be treated as reliable elephant locations.</p>"
    if (output/"controls.json").exists():
        control = json.loads((output/"controls.json").read_text())
        findings += '<h2>Geometry and initialization controls</h2><p>Windows selected using fitting error only. Restart displacement is sensitivity, not an accuracy bound.</p><table><tr><th>Start (s)</th><th>Original held-out NMSE</th><th>Restart held-out</th><th>Shuffled held-out</th><th>Restart location changes (m)</th></tr>'
        for row in control["controls"]:
            findings += (f'<tr><td>{row["window_start_s"]:g}</td><td>{row["original_heldout_nmse"]:.3f}</td>'
                         f'<td>{row["restart"]["heldout_nmse"]:.3f}</td><td>{row["shuffled_geometry"]["heldout_nmse"]:.3f}</td>'
                         f'<td>{", ".join(f"{d:.1f}" for d in row["restart"]["matched_location_change_m"])}</td></tr>')
        findings += '</table><p><a href="controls.json">Full controls</a>. One shuffle per case is a diagnostic, not a statistical test.</p>'
    if (output/"injection.json").exists():
        injection = json.loads((output/"injection.json").read_text())
        findings += '<h2>Known calls injected into measured background</h2><p>Two overlapping harmonic calls at +12 dB total band power relative to background, known K/duration and matched propagation. This favorable control does not measure real-elephant accuracy.</p><ul>'
        for i,row in enumerate(injection["sources"]):
            findings += (f'<li>Call {i+1}: position error {row["position_error_m"]:.1f} m; '
                         f'M04 SI-SDR {row["mixture_si_sdr_db"]:.1f} → {row["separated_si_sdr_db"]:.1f} dB.</li>')
        findings += '</ul><img src="injection/evaluation.png"><p><a href="injection.json">Injection metrics</a>. <a href="injection/mixture_4x.wav">Mixture 4×</a>, <a href="injection/separated_1_4x.wav">call 1</a>, <a href="injection/separated_2_4x.wav">call 2</a>.</p>'
    rows = "".join(f'<tr><td><a href="{r["directory"]}/reconstruction.png">{r["window_start_s"]:g}</a></td>'
                   f'<td>{r["K"]}</td><td>{r["fit_nmse"]:.3f}</td><td>{r["heldout_nmse"]:.3f}</td>'
                   f'<td><a href="{r["directory"]}/map.png">map</a> <a href="tdoa_{r["window_start_s"]:06.2f}.png">waveform map</a></td>'
                   f'<td><audio controls preload="none" src="{r["directory"]}/mixture_4x.wav"></audio></td>'
                   f'<td>' + "".join(f'<audio controls preload="none" src="{r["directory"]}/component_{k+1}_4x.wav"></audio>' for k in range(r["K"])) + '</td></tr>'
                   for r in results)
    document = f'''<!doctype html><meta charset="utf-8"><title>Real rumble POC</title>
<style>body{{font:16px system-ui;margin:30px;max-width:1500px}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}audio{{width:210px}}</style>
<h1>Real multichannel rumble feasibility experiment</h1>
<p>{protocol['arguments']['window']:g}-second overlapping windows, a fresh shared neural Gaussian decoder per fit,
fixed source-count priors, and two stations excluded from fitting. Candidates are unlabelled
low-frequency acoustic components. There is no ground truth for elephant separation or localization.</p>
<p>Fit stations: {', '.join(protocol['fit_stations'])}. Held out: {', '.join(protocol['heldout_stations'])}.
Background is a recording-wide 20th percentile per station/bin. Lower NMSE is better; 1 is zero excess prediction.</p>
{findings}<img src="summary.png"><details><summary>Station geometry and recording overview</summary><img src="overview.png"></details>
<p>Independent 20–110 Hz waveform pair-correlation maps use the same six fitting stations. These show the strongest coherent spatial peak per window; broad or multiple peaks are not a calibrated location posterior. See <a href="tdoa.json">waveform diagnostics</a>.</p>
<p>Audio below plays 4× faster (frequency and time shifted); original-rate WAVs are in each case folder.
All components and residual use the same gain as their mixture. Listen for leakage and splitting of a single call.</p>
<table><tr><th>Window (s)</th><th>K</th><th>Fit NMSE</th><th>Held-out NMSE</th><th>Location</th><th>Mixture 4×</th><th>Components 4×</th></tr>{rows}</table>
<h2>Interpretation limits</h2><ul>{caveats}</ul>
<p>Artifacts: <a href="protocol.json">protocol</a>, <a href="results.json">metrics</a>,
<a href="candidates.csv">candidate catalog</a>. Catalog repeats overlapping windows; it does not count individual elephants.</p>'''
    (output/"report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("real_world/source_data/elephant rumble/multi_aud_20230308_104500.wav"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/real_rumbles"))
    parser.add_argument("--anchor", default="M04")
    parser.add_argument("--stations", type=int, choices=(6,7,8), default=8)
    parser.add_argument("--window", type=float, default=8.)
    parser.add_argument("--hop", type=float, default=4.)
    parser.add_argument("--call-duration", type=float, default=5.)
    parser.add_argument("--low-hz", type=float, default=12.)
    parser.add_argument("--high-hz", type=float, default=120.)
    parser.add_argument("--margin", type=float, default=250.)
    parser.add_argument("--distance-offset", type=float, default=50.)
    parser.add_argument("--events", type=int, nargs="+", choices=(1,2,3,4,5), default=[1,2])
    parser.add_argument("--starts", type=float, nargs="+")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--splauts", type=int, default=48)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--report-only", action="store_true", help="Regenerate HTML after post-fit controls")
    args = parser.parse_args()
    if args.report_only:
        make_report(args.output, json.loads((args.output/"protocol.json").read_text()),
                    json.loads((args.output/"results.json").read_text()), [])
        return
    if (min(args.window,args.hop,args.call_duration,args.distance_offset,args.margin,args.threads,args.splauts) <= 0
            or args.call_duration >= args.window or args.hop > args.window or args.grid_size < 2
            or min(args.steps,args.warmup_steps) < 0 or not 0 < args.low_hz < args.high_hz < 256):
        parser.error("Invalid window, band, geometry or optimization settings")
    run(args)


if __name__ == "__main__":
    main()
