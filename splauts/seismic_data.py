"""Dense seismic-array selection and bounded-memory spectral preparation."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .real_data import StationTable, load_audio, stft_power


def select_stations(metadata: Path, channels: int, radius: float = 500.,
                    fit_count: int = 16, validation_count: int = 16, seed: int = 2026,
                    origin=(.4560107, 36.8429853)):
    """Use explicit one-based tracks; never invent channels absent from the WAV."""
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    tracks = np.array([int(r["TrackIndex"])-1 for r in rows])
    ll = np.array([[float(r["Lat"]), float(r["Lon"])] for r in rows])
    if len(np.unique(tracks)) != len(tracks) or (tracks < 0).any() or not np.isfinite(ll).all():
        raise ValueError("Metadata requires unique positive track indices and finite coordinates")
    origin = np.asarray(origin)
    xy = (ll[:,::-1]-origin[::-1]) * (np.pi/180*6371000)
    xy[:,0] *= np.cos(np.deg2rad(origin[0]))
    nearby = np.linalg.norm(xy, axis=1) <= radius
    eligible = np.flatnonzero(nearby & (tracks < channels))
    if not 10 <= fit_count <= 20 or validation_count < 1 or len(eligible) <= fit_count+validation_count:
        raise ValueError("Require 10–20 fitting stations and enough distinct validation/test stations")
    table = StationTable([rows[i]["StationID"] for i in eligible], tracks[eligible],
                         xy[eligible], ll[eligible], origin)
    order = np.random.default_rng(seed).permutation(len(eligible))
    fit = np.sort(order[:fit_count])
    validation = np.sort(order[fit_count:fit_count+validation_count])
    test = np.sort(order[fit_count+validation_count:])
    if np.linalg.matrix_rank(table.xy[fit]-table.xy[fit].mean(0)) < 2:
        raise ValueError("Random fitting stations have degenerate geometry")
    provenance = {"csv_rows": len(rows), "wav_channels": channels,
                  "metadata_tracks_absent_from_wav": (tracks[tracks>=channels]+1).tolist(),
                  "nearby_metadata_tracks_absent_from_wav": (tracks[nearby & (tracks>=channels)]+1).tolist(),
                  "mapping_assumption": "Available CSV TrackIndex values refer directly to WAV channels; absent tracks are excluded. This does not verify export ordering.",
                  "radius_m": radius, "origin_lat_lon": origin.tolist(), "split_seed": seed}
    return table, {"fit":fit, "validation":validation, "test":test}, provenance


def prepare(audio_path: Path, metadata: Path, directory: Path, radius=500.,
            fit_count=16, validation_count=16, seed=2026, low=8., high=80.):
    directory.mkdir(parents=True, exist_ok=True)
    info = sf.info(audio_path)
    if info.samplerate != 500:
        raise ValueError("This seismic protocol requires native 500 Hz data")
    table, splits, mapping = select_stations(metadata, info.channels, radius, fit_count, validation_count, seed)
    signature = {"audio":str(audio_path.resolve()), "size":audio_path.stat().st_size,
                 "mtime_ns":audio_path.stat().st_mtime_ns, "metadata":str(metadata.resolve()),
                 "metadata_mtime_ns":metadata.stat().st_mtime_ns, "radius":radius,
                 "fit_count":fit_count,"validation_count":validation_count,"seed":seed,"band":[low,high]}
    protocol_path = directory/"data_protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if old["signature"] != signature:
            raise ValueError("Prepared data settings changed; use another --data directory")
        return table, splits, old
    audio, wav = load_audio(audio_path, table, sample_rate=500)
    np.save(directory/"audio.npy", audio)
    all_frequencies = np.fft.rfftfreq(500, 1/500)
    band = (all_frequencies>=low)&(all_frequencies<=high)
    power, reference_z, times = [], None, None
    # The reference belongs to fitting data, selected using coordinates only.
    reference = int(splits["fit"][np.linalg.norm(table.xy[splits["fit"]],axis=1).argmin()])
    for first in range(0,len(audio),16):
        last = min(first+16,len(audio))
        z,times,_ = stft_power(audio[first:last],500,500,50)
        power.append((np.abs(z[...,band])**2).astype(np.float32))
        if first <= reference < last:
            reference_z = z[reference-first].copy()
    raw = np.concatenate(power)
    floor = np.quantile(raw,.2,axis=1,keepdims=True).astype(np.float32)
    # A common scale is estimated using fitting channels only.
    scale = float(np.quantile(raw[splits["fit"]],.99))
    if scale <= 0 or np.any(np.mean(raw**2,axis=(1,2)) <= 0):
        raise ValueError("Empty channels in the spatial subset")
    np.save(directory/"power.npy", np.maximum(raw-floor,0)/scale)
    np.savez(directory/"axes.npz", times=times, frequencies=all_frequencies[band],
             background=floor/scale, reference_z=reference_z/np.sqrt(scale), band=band)
    protocol = {"signature":signature,"wav":wav,"mapping":mapping,
                "stations":[{"id":label,"track_index":int(track+1),"east_north_m":xy.tolist(),
                             "lat_lon":ll.tolist()} for label,track,xy,ll in zip(table.ids,table.tracks,table.xy,table.latlon)],
                "splits":{k:v.tolist() for k,v in splits.items()},"reference_index":reference,
                "reference_station":table.ids[reference],"power_scale":scale,
                "preprocessing":"Native 500 Hz; 1 s Hann STFT; 0.1 s hop; recording-wide per-channel/bin 20th percentile floor; one fit-only common scale.",
                "heldout_calibration":"Validation/test observations estimate their own stationary noise floors only; their transient powers never enter source fitting or initialization."}
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    overview(directory,table,splits,raw,times,all_frequencies[band],reference)
    return table,splits,protocol


def overview(directory,table,splits,power,times,frequencies,reference):
    fig,axes=plt.subplots(2,1,figsize=(13,9),constrained_layout=True)
    colors={"fit":"tab:orange","validation":"tab:green","test":"tab:blue"}
    for key,indices in splits.items():
        axes[0].scatter(*table.xy[indices].T,s=40 if key=="fit" else 12,
                        marker="^" if key=="fit" else ".",color=colors[key],label=f"{key}: {len(indices)}")
    axes[0].scatter(0,0,marker="x",color="black",label="M04 acoustic reference")
    axes[0].set(xlabel="East of M04 (m)",ylabel="North of M04 (m)",aspect="equal")
    axes[0].legend()
    mean=np.median(power[splits["fit"]],axis=0)
    vmax=float(np.quantile(mean,.999))
    axes[1].pcolormesh(times,frequencies,10*np.log10(np.maximum(mean.T,vmax*1e-6)/vmax),
                       shading="auto",vmin=-45,vmax=0)
    axes[1].set(xlabel="Seconds from WAV start",ylabel="Hz",title="Median spectral power across fitting stations (before floor subtraction)")
    fig.savefig(directory/"overview.png",dpi=140)
    plt.close(fig)
    chosen=splits["fit"]
    fig,axes=plt.subplots(8,2,figsize=(15,17),constrained_layout=True)
    for ax,i in zip(axes.flat,chosen):
        vmax=float(np.quantile(power[i],.999))
        ax.pcolormesh(times,frequencies,10*np.log10(np.maximum(power[i].T,vmax*1e-6)/vmax),
                      shading="auto",vmin=-40,vmax=0)
        ax.set(title=table.ids[i],ylabel="Hz")
    for ax in axes[-1]:
        ax.set_xlabel("Seconds")
    fig.savefig(directory/"fit_station_spectrograms.png",dpi=110)
    plt.close(fig)
