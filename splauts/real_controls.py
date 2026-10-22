"""Post-fit geometry/restart controls; training data alone chooses audit windows."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy import signal
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .real_experiments import render_components, score
from .real_data import stft_power, audio_from_mask
from .training.mixtures import MixtureObservations, fit_mixtures


def audit(directory: Path, cases: int = 3):
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    protocol = json.loads((directory/"protocol.json").read_text())
    results = json.loads((directory/"results.json").read_text())
    args = protocol["arguments"]
    config = Config(**protocol["config"])
    candidates = sorted((r for r in results if r["K"] == max(args["events"])), key=lambda r:r["fit_nmse"])
    selected = []
    for row in candidates:
        if all(abs(row["window_start_s"]-r["window_start_s"]) >= args["window"] for r in selected):
            selected.append(row)
        if len(selected) == cases:
            break
    output = []
    for row in selected:
        saved = torch.load(directory/row["directory"]/"fit.pt", weights_only=True)
        train, held = saved["train_indices"], saved["heldout_indices"]
        stations, times, frequencies = saved["stations"], saved["times"], saved["frequencies"]
        power = saved["observation"]
        core = ((times >= row["window_start_s"]-saved["audio_time_origin_s"])
                & (times < row["window_start_s"]+args["window"]-saved["audio_time_origin_s"]))
        entry = {"window_start_s": row["window_start_s"], "K": row["K"],
                 "original_fit_nmse": row["fit_nmse"], "original_heldout_nmse": row["heldout_nmse"]}
        for label in ("restart", "shuffled_geometry"):
            coords = stations[train].clone()
            if label == "shuffled_geometry":
                # Keep the same coordinate set and six mixtures, but break their association.
                coords = coords.roll(2, dims=0)
            observed = MixtureObservations(power[train], coords, times, frequencies.float())
            fit = fit_mixtures(observed, config, row["K"], args["warmup_steps"], args["steps"],
                               geometry_steps=0, proposal_grid_size=args["grid_size"],
                               fit_seed=args["seed"]+(1 if label == "restart" else 0),
                               bandwidth_hz=args["high_hz"]-args["low_hz"], log_every=0)
            fit_prediction = render_components(fit, coords, times, frequencies.float(), config).sum(0)
            held_prediction = render_components(fit, stations[held], times, frequencies.float(), config).sum(0)
            distances = torch.cdist(saved["state"]["positions"], fit.state.positions)
            order = min(itertools.permutations(range(row["K"])), key=lambda p:sum(float(distances[k,p[k]]) for k in range(row["K"])))
            entry[label] = {"fit_nmse": score(fit_prediction[:,core], power[train][:,core]),
                            "heldout_nmse": score(held_prediction[:,core], power[held][:,core]),
                            "matched_location_change_m": [float(distances[k,order[k]]) for k in range(row["K"])],
                            "positions": fit.state.positions.tolist()}
        print(json.dumps(entry), flush=True)
        output.append(entry)
    summary = {"selection": "Lowest fit-station NMSE K=max windows separated by >=window duration; no held-out selection",
               "controls": output,
               "interpretation": "Restart displacement is sensitivity, not an accuracy bound. Geometry shuffle is one permutation per case, not a statistical test."}
    (directory/"controls.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def injection(directory: Path):
    """Known overlapping rumble-like waveforms added to measured field background.

    Sources are synthesized independently in waveform space; only their mixtures
    enter fitting. This is a favorable controlled check, not a real-event score.
    """
    protocol = json.loads((directory/"protocol.json").read_text())
    config = Config(**protocol["config"])
    args = protocol["arguments"]
    with np.load(directory/"audio_cache.npz") as cache:
        background = cache["audio"][:, :16*512]
    xy = np.array([r["east_north_m"] for r in protocol["stations"]])
    origin = (xy.min(0)+xy.max(0)-config.area_size)/2
    stations = torch.as_tensor(xy-origin, dtype=torch.float32)
    labels = [r["id"] for r in protocol["stations"]]
    train = np.array([labels.index(s) for s in protocol["fit_stations"]])
    held = np.setdiff1d(np.arange(len(xy)), train)
    source_xy = np.array([[-80., -200.], [180., 160.]])
    t = np.arange(background.shape[1])/512
    contributions = []
    for k, (position, f0, onset) in enumerate(zip(source_xy, [26.,39.], [2.,4.])):
        distance = np.linalg.norm(xy-position, axis=1)
        u = t[None]-onset-distance[:,None]/config.sound_speed
        phase = 2*np.pi*(f0*u + 2.0*config.call_duration/np.pi*(1-np.cos(np.pi*u/config.call_duration)))
        envelope = np.where((u>0)&(u<config.call_duration), np.sin(np.pi*u/config.call_duration)**2, 0)
        waveform = envelope*sum(np.sin(h*phase+.3*k)/h for h in (1,2,3))
        contributions.append(waveform/(distance[:,None]+config.distance_offset))
    contributions = np.array(contributions)
    bandpass = signal.butter(4,[12,120],btype="bandpass",fs=512,output="sos")
    noise = signal.sosfiltfilt(bandpass, background, axis=1)
    common_gain = np.sqrt(np.mean(noise[train]**2)*10**(12/10)/np.mean(contributions[:,train].sum(0)**2))
    contributions *= common_gain
    mixture = background+contributions.sum(0)
    z, times, frequencies = stft_power(mixture)
    band = (frequencies>=args["low_hz"])&(frequencies<=args["high_hz"])
    raw = np.abs(z[...,band])**2
    floor = np.quantile(raw,.2,axis=1,keepdims=True)
    scale = np.quantile(raw,.99)
    power = torch.as_tensor(np.maximum(raw-floor,0)/scale,dtype=torch.float32)
    obs = MixtureObservations(power[train],stations[train],torch.as_tensor(times,dtype=torch.float32),
                              torch.as_tensor(frequencies[band].copy(),dtype=torch.float32))
    fit = fit_mixtures(obs,config,2,200,400,geometry_steps=0,proposal_grid_size=args["grid_size"],
                       fit_seed=args["seed"],bandwidth_hz=args["high_hz"]-args["low_hz"],log_every=100)
    components = render_components(fit,stations,obs.times,obs.frequencies,config)
    estimates = fit.state.positions.numpy()+origin
    distance = np.linalg.norm(source_xy[:,None]-estimates[None],axis=-1)
    order = min(itertools.permutations(range(2)),key=lambda p:sum(distance[k,p[k]] for k in range(2)))
    denom = np.maximum(raw[0],components[:,0].sum(0).numpy()*scale+floor[0])
    source_metrics = []
    separated = []
    for k, estimated in enumerate(order):
        true_z, _, _ = stft_power(contributions[k,0:1])
        true_power = np.abs(true_z[0,:,band].T)**2/scale
        predicted = components[estimated,0].numpy()
        nmse = float(np.mean((predicted-true_power)**2)/np.mean(true_power**2))
        mask = np.zeros_like(np.abs(z[0])**2)
        mask[:,band] = components[estimated,0].numpy()*scale/np.maximum(denom,1e-20)
        wave = audio_from_mask(z[0],mask)[:len(t)]
        separated.append(wave)
        truth = contributions[k,0]
        def si_sdr(estimate):
            aligned = np.dot(estimate,truth)/max(np.dot(truth,truth),1e-20)*truth
            return float(10*np.log10(max(np.sum(aligned**2),1e-20)/max(np.sum((estimate-aligned)**2),1e-20)))
        band_mask = np.broadcast_to(band, mask.shape)
        ref_mixture = audio_from_mask(z[0],band_mask)[:len(t)]
        source_metrics.append({"true_east_north_m":source_xy[k].tolist(),"estimated_east_north_m":estimates[estimated].tolist(),
                               "position_error_m":float(distance[k,estimated]),"reference_source_power_nmse":nmse,
                               "mixture_si_sdr_db":si_sdr(ref_mixture),"separated_si_sdr_db":si_sdr(wave)})
    report = {"description":"Two independently synthesized overlapping 5 s harmonic rumble-like calls on first 16 s of measured background",
              "injected_total_to_background_band_power_db":12.,"fit_stations":protocol["fit_stations"],
              "fit_nmse":score(components.sum(0)[train],power[train]),
              "heldout_nmse":score(components.sum(0)[held],power[held]),"sources":source_metrics,
              "caveat":"Known K/duration, strong injected sources and matched propagation assumptions; not real-elephant accuracy"}
    (directory/"injection.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    artifacts = directory/"injection"
    artifacts.mkdir(exist_ok=True)
    gain = .8/max(float(np.abs(mixture[0]).max()),1e-12)
    for label,wave in [("mixture",mixture[0]), *[(f"truth_{k+1}",w[0]) for k,w in enumerate(contributions)],
                       *[(f"separated_{k+1}",w) for k,w in enumerate(separated)]]:
        sf.write(artifacts/f"{label}.wav",wave*gain,512,subtype="FLOAT")
        sf.write(artifacts/f"{label}_4x.wav",wave*gain,2048,subtype="FLOAT")
    torch.save({"config":protocol["config"],"generator":fit.generator.state_dict(),"state":vars(fit.state),
                "splauts":vars(fit.splauts),"true_positions":torch.as_tensor(source_xy),
                "true_received_waveforms":torch.as_tensor(contributions),"mixture_waveforms":torch.as_tensor(mixture),
                "predicted_components":components,"stations":stations,"times":obs.times,"frequencies":obs.frequencies}, artifacts/"fit.pt")
    fig,axes = plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    axes[0].scatter(*xy.T,marker="^",label="stations")
    axes[0].scatter(*source_xy.T,marker="o",label="injected truth")
    axes[0].scatter(*estimates.T,marker="x",label="inferred")
    axes[0].set(xlabel="East of M04 (m)",ylabel="North of M04 (m)",aspect="equal")
    axes[0].legend()
    x=np.arange(2)
    axes[1].bar(x-.18,[r["mixture_si_sdr_db"] for r in source_metrics],.36,label="mixture")
    axes[1].bar(x+.18,[r["separated_si_sdr_db"] for r in source_metrics],.36,label="separated")
    axes[1].set(xticks=x,xticklabels=["Injected call 1","Injected call 2"],ylabel="SI-SDR at M04 (dB)")
    axes[1].legend()
    fig.savefig(artifacts/"evaluation.png",dpi=130)
    plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/real_rumbles"))
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--injection", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    if args.injection:
        injection(args.output)
    else:
        audit(args.output, args.cases)
