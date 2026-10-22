"""Blind injection and geometry controls for the dense seismic experiment."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import soundfile as sf
import torch

from .config import Config
from .model.renderer import render
from .real_data import stft_power, audio_from_mask
from .real_experiments import render_components
from .real_tdoa import spatial_correlation
from .seismic_experiments import station_nmse
from .training.mixtures import MixtureObservations, fit_mixtures


def load(directory):
    protocol=json.loads((directory/"protocol.json").read_text())
    data_path=Path(protocol["data"])
    data=json.loads((data_path/"data_protocol.json").read_text())
    xy=np.array([s["east_north_m"] for s in data["stations"]])
    positions=torch.as_tensor(xy-np.array(protocol["search_origin_east_north_m"]),dtype=torch.float32)
    splits={k:np.array(v) for k,v in data["splits"].items()}
    return protocol,data_path,data,xy,positions,splits


def geometry(directory,starts=(28.,36.,44.)):
    protocol,path,data,xy,positions,splits=load(directory)
    args=protocol["arguments"]
    config=Config(**protocol["config"])
    rows=[]
    for start in starts:
        case=directory/f"window_{start:06.2f}_K2"
        if not (case/"fit.pt").exists():
            continue
        saved=torch.load(case/"fit.pt",weights_only=True)
        original=json.loads((case/"metrics.json").read_text())
        times,frequencies=saved["times"],saved["frequencies"]
        first=round(saved["time_origin"]*10)
        power=torch.as_tensor(np.array(np.load(path/"power.npy",mmap_mode="r")[:,first:first+len(times)]))
        row={"window_start_s":start,"original":original["metrics"],"controls":{}}
        for label in ("restart","shuffled_geometry"):
            coords=positions[splits["fit"]].clone()
            if label=="shuffled_geometry":
                coords=coords[torch.as_tensor(np.random.default_rng(303).permutation(len(coords)))]
            observed=MixtureObservations(power[splits["fit"]],coords,times,frequencies)
            fit=fit_mixtures(observed,config,2,args["warmup_steps"],args["steps"],geometry_steps=0,
                               proposal_grid_size=args["grid_size"],fit_seed=args["seed"]+(label=="restart"),
                               bandwidth_hz=args["high_hz"]-args["low_hz"],log_every=0,time_mask=saved["time_mask"])
            errors=[]
            for first_station in range(0,len(positions),16):
                last=min(first_station+16,len(positions))
                with torch.no_grad():
                    coords_eval=positions[first_station:last].clone()
                    # Fit error must evaluate the shuffled station association, too.
                    for local,i in enumerate(range(first_station,last)):
                        found=np.flatnonzero(splits["fit"]==i)
                        if len(found): coords_eval[local]=coords[int(found[0])]
                    prediction=render(fit.splauts,fit.state.positions,fit.state.onsets,fit.state.amplitudes,
                                      coords_eval,times,frequencies,config)
                    core=saved["time_mask"]
                    errors.extend(station_nmse(prediction[:,core],power[first_station:last,core]).tolist())
            errors=np.array(errors)
            distance=torch.cdist(saved["state"]["positions"],fit.state.positions)
            order=min(itertools.permutations(range(2)),key=lambda p:sum(float(distance[k,p[k]]) for k in range(2)))
            row["controls"][label]={"mean_nmse":{k:float(errors[v].mean()) for k,v in splits.items()},
                                     "matched_position_change_m":[float(distance[k,order[k]]) for k in range(2)]}
        print(json.dumps(row),flush=True)
        rows.append(row)
    result={"selection":"Fixed 28/36/44 s windows from visible harmonic contours in fitting channels; no test-based selection",
            "caveat":"One coordinate permutation and one decoder-seed restart per case; sensitivity diagnostic, not a localization confidence interval", "windows":rows}
    (directory/"controls.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


def injection(directory):
    protocol,path,data,xy,positions,splits=load(directory)
    config=Config(**protocol["config"])
    background=np.array(np.load(path/"audio.npy",mmap_mode="r")[:,:8000])
    t=np.arange(8000)/500
    truth_positions=np.array([[50.,-150.],[-180.,180.]])
    sources=[]
    for k,(point,f0,onset) in enumerate(zip(truth_positions,[19.,27.],[2.,4.])):
        distance=np.linalg.norm(xy-point,axis=1)
        u=t[None]-onset-distance[:,None]/330
        phase=2*np.pi*(f0*u+2*config.call_duration/np.pi*(1-np.cos(np.pi*u/config.call_duration)))
        envelope=np.where((u>0)&(u<config.call_duration),np.sin(np.pi*u/config.call_duration)**2,0)
        wave=envelope*sum(np.sin(h*phase+.37*k)/h for h in (1,2,3))
        sources.append(wave/(distance[:,None]+config.distance_offset)**(config.power_spreading_exponent/2))
    sources=np.array(sources)
    noise=signal.sosfiltfilt(signal.butter(4,[8,80],fs=500,btype="bandpass",output="sos"),background,axis=1)
    gain=np.sqrt(np.mean(noise[splits["fit"]]**2)*10**1.2/np.mean(sources[:,splits["fit"]].sum(0)**2))
    sources*=gain
    mixture=background+sources.sum(0)
    frequencies=np.fft.rfftfreq(500,1/500)
    band=(frequencies>=8)&(frequencies<=80)
    powers=[]
    reference=data["reference_index"]
    z_reference=None
    for first in range(0,len(mixture),16):
        z,times,_=stft_power(mixture[first:first+16],500,500,50)
        powers.append(np.abs(z[...,band])**2)
        if first<=reference<first+16: z_reference=z[reference-first]
    raw=np.concatenate(powers)
    floor=np.quantile(raw,.2,axis=1,keepdims=True)
    scale=float(np.quantile(raw[splits["fit"]],.99))
    power=torch.as_tensor(np.maximum(raw-floor,0)/scale,dtype=torch.float32)
    times=torch.as_tensor(times,dtype=torch.float32)
    frequencies_t=torch.as_tensor(frequencies[band],dtype=torch.float32)
    core=(times>=1)&(times<13)
    observed=MixtureObservations(power[splits["fit"]],positions[splits["fit"]],times,frequencies_t)
    fit=fit_mixtures(observed,config,2,200,400,geometry_steps=0,
                       proposal_grid_size=protocol["arguments"]["grid_size"],fit_seed=101,
                       bandwidth_hz=72.,log_every=100,time_mask=core)
    errors=[]
    for first in range(0,len(mixture),16):
        with torch.no_grad():
            prediction=render(fit.splauts,fit.state.positions,fit.state.onsets,fit.state.amplitudes,
                              positions[first:first+16],times,frequencies_t,config)
            errors.extend(station_nmse(prediction[:,core],power[first:first+16,core]).tolist())
    errors=np.array(errors)
    origin=np.array(protocol["search_origin_east_north_m"])
    estimates=fit.state.positions.numpy()+origin
    distance=np.linalg.norm(truth_positions[:,None]-estimates[None],axis=-1)
    order=min(itertools.permutations(range(2)),key=lambda p:sum(distance[k,p[k]] for k in range(2)))
    received=render_components(fit,positions[reference:reference+1],times,frequencies_t,config)[:,0].numpy()*scale
    denominator=np.maximum(np.abs(z_reference[:,band])**2,received.sum(0)+floor[reference])
    waves=[]
    metrics=[]
    full=np.broadcast_to(band,z_reference.shape)
    ref_mixture=audio_from_mask(z_reference,full,500,500,50)[:8000]
    def si_sdr(estimate,truth):
        aligned=np.dot(estimate,truth)/max(np.dot(truth,truth),1e-20)*truth
        return float(10*np.log10(max(np.sum(aligned**2),1e-20)/max(np.sum((estimate-aligned)**2),1e-20)))
    for k,match in enumerate(order):
        mask=np.zeros(z_reference.shape)
        mask[:,band]=received[match]/np.maximum(denominator,1e-20)
        wave=audio_from_mask(z_reference,mask,500,500,50)[:8000]
        waves.append(wave)
        true_z,_,_=stft_power(sources[k,reference:reference+1],500,500,50)
        true_power=np.abs(true_z[0][:,band])**2
        metrics.append({"true_east_north_m":truth_positions[k].tolist(),"estimated_east_north_m":estimates[match].tolist(),
                        "position_error_m":float(distance[k,match]),
                        "reference_power_nmse":float(np.mean((received[match]-true_power)**2)/np.mean(true_power**2)),
                        "mixture_si_sdr_db":si_sdr(ref_mixture,sources[k,reference]),
                        "separated_si_sdr_db":si_sdr(wave,sources[k,reference])})
    result={"description":"Two independent overlapping harmonic waveform calls injected into first 16 s of measured seismic background",
            "injected_total_band_snr_db":12.,"reference":data["reference_station"],"sources":metrics,
            "nmse":{k:{"mean":float(errors[v].mean()),"median":float(np.median(errors[v]))} for k,v in splits.items()},
            "caveat":"Favorable known-count/duration, strong-source, matched-330m/s-and-spreading control; not native elephant ground truth"}
    (directory/"injection.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    artifacts=directory/"injection"
    artifacts.mkdir(exist_ok=True)
    audio_gain=.8/max(float(np.abs(ref_mixture).max()),1e-12)
    for label,wave in [("mixture",ref_mixture),*[(f"truth_{k+1}",s[reference]) for k,s in enumerate(sources)],*[(f"separated_{k+1}",w) for k,w in enumerate(waves)]]:
        sf.write(artifacts/f"{label}.wav",wave*audio_gain,500,subtype="FLOAT")
        sf.write(artifacts/f"{label}_4x.wav",wave*audio_gain,2000,subtype="FLOAT")
    torch.save({"config":asdict(config),"state":asdict(fit.state),"splauts":asdict(fit.splauts),
                "generator":fit.generator.state_dict(),"fit_observation":observed.power,"fit_stations":observed.stations,
                "times":times,"frequencies":frequencies_t,"source_reference_waves":torch.as_tensor(sources[:,reference]),
                "true_positions":torch.as_tensor(truth_positions),"estimated_positions":torch.as_tensor(estimates)},artifacts/"fit.pt")
    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axs[0].scatter(*xy[splits["test"]].T,s=8,label="test stations")
    axs[0].scatter(*xy[splits["fit"]].T,marker="^",label="fit")
    axs[0].scatter(*truth_positions.T,marker="o",label="injected truth")
    axs[0].scatter(*estimates.T,marker="x",s=100,label="inferred")
    axs[0].set(xlabel="East of M04 (m)",ylabel="North of M04 (m)",aspect="equal")
    axs[0].legend(fontsize=8)
    x=np.arange(2)
    axs[1].bar(x-.18,[m["mixture_si_sdr_db"] for m in metrics],.36,label="mixture")
    axs[1].bar(x+.18,[m["separated_si_sdr_db"] for m in metrics],.36,label="separated")
    axs[1].set(xticks=x,xticklabels=["Injected call 1","Injected call 2"],ylabel="Reference SI-SDR (dB)")
    axs[1].legend()
    fig.savefig(artifacts/"evaluation.png",dpi=130)
    plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


def waveform_maps(directory):
    protocol,path,data,xy,positions,splits=load(directory)
    rows=json.loads((directory/"selected.json").read_text())
    audio=np.array(np.load(path/"audio.npy",mmap_mode="r")[splits["fit"]])
    audio=signal.sosfiltfilt(signal.butter(4,[12,70],btype="bandpass",fs=500,output="sos"),audio,axis=1)
    origin=np.array(protocol["search_origin_east_north_m"])
    area=protocol["config"]["area_size"]
    east=np.linspace(origin[0],origin[0]+area,81)
    north=np.linspace(origin[1],origin[1]+area,81)
    yy,xx=np.meshgrid(north,east,indexing="ij")
    grid=np.stack([xx.ravel(),yy.ravel()],axis=-1)
    summary=[]
    for row in rows:
        start=row["window_start_s"]
        values,pairs=spatial_correlation(audio[:,round(start*500):round((start+protocol["arguments"]["window"])*500)],xy[splits["fit"]],grid,500,330.)
        peak=int(values.argmax())
        summary.append({"window_start_s":start,"peak_east_north_m":grid[peak].tolist(),
                        "mean_pair_correlation":float(values[peak]),"pairs":pairs})
        fig,ax=plt.subplots(figsize=(7,7),constrained_layout=True)
        im=ax.imshow(values.reshape(81,81),origin="lower",extent=[east[0],east[-1],north[0],north[-1]],cmap="viridis")
        ax.scatter(*xy[splits["fit"]].T,marker="^",color="white",label="fitting stations")
        ax.scatter(*grid[peak],marker="x",s=100,color="red",label="waveform peak")
        points=np.array([[c["east_m"],c["north_m"]] for c in row["candidates"]])
        ax.scatter(*points.T,marker="*",s=120,color="orange",label="validation-selected splauts")
        ax.set(title=f"{start:g} s: 12–70 Hz waveform correlation",xlabel="East of M04 (m)",ylabel="North of M04 (m)")
        ax.legend(fontsize=8)
        fig.colorbar(im,ax=ax,label="Mean normalized waveform correlation")
        fig.savefig(directory/f"tdoa_{start:06.2f}.png",dpi=110)
        plt.close(fig)
    (directory/"tdoa.json").write_text(json.dumps({"method":"Normalized raw waveform pair correlation; fitting stations only; one maximum per window", "windows":summary},indent=2),encoding="utf-8")
    print(f"Saved {len(summary)} waveform maps",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("outputs/seismic_rumbles"))
    p.add_argument("--injection",action="store_true")
    p.add_argument("--waveform-maps",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    if args.injection: injection(args.output)
    elif args.waveform_maps: waveform_maps(args.output)
    else: geometry(args.output)
