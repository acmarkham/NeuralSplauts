"""Windowed neural-splaut localization with dense seismic station holdouts."""

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
import soundfile as sf
import torch

from .config import Config
from .model.generator import Splauts, evaluate_field
from .model.renderer import render
from .real_data import audio_from_mask, window_starts
from .real_experiments import render_components
from .seismic_data import prepare
from .training.mixtures import MixtureObservations, fit_mixtures


def station_nmse(prediction, target):
    return ((prediction-target).square().mean((1,2)) /
            target.square().mean((1,2)).clamp_min(1e-20))


def interpolate_stations(fit_power, fit_xy, query_xy):
    """Fit-data-only inverse-distance interpolation; no held-out gain fitting."""
    distance=torch.cdist(query_xy,fit_xy)
    nearest=distance.topk(min(3,len(fit_xy)),largest=False)
    weights=(nearest.values+25).pow(-2)
    weights=weights/weights.sum(1,keepdim=True)
    return (fit_power[nearest.indices]*weights[:,:,None,None]).sum(1)


def evaluate(fit,power,positions,times,frequencies,config,splits,core):
    """Render held-out sensors in bounded batches, retaining errors rather than huge tensors."""
    errors,baseline=[],[]
    diagnostic=[]
    chosen=np.r_[splits["fit"][:3],splits["test"][:5]]
    fit_power=power[splits["fit"]]
    for first in range(0,len(power),16):
        last=min(first+16,len(power))
        with torch.no_grad():
            prediction=render(fit.splauts,fit.state.positions,fit.state.onsets,fit.state.amplitudes,
                              positions[first:last],times,frequencies,config)
            errors.extend(station_nmse(prediction[:,core],power[first:last,core]).tolist())
            simple=interpolate_stations(fit_power,positions[splits["fit"]],positions[first:last])
            baseline.extend(station_nmse(simple[:,core],power[first:last,core]).tolist())
            for i in chosen:
                if first<=i<last:
                    diagnostic.append((int(i),prediction[i-first].clone()))
    errors=np.array(errors)
    baseline=np.array(baseline)
    metrics={}
    for key,indices in splits.items():
        metrics[key]={"count":len(indices),"mean_nmse":float(errors[indices].mean()),
                      "median_nmse":float(np.median(errors[indices])),
                      "p90_nmse":float(np.quantile(errors[indices],.9)),
                      "below_zero_fraction":float(np.mean(errors[indices]<1)),
                      "interpolation_mean_nmse":float(baseline[indices].mean()),
                      "interpolation_median_nmse":float(np.median(baseline[indices])),
                      "beats_interpolation_fraction":float(np.mean(errors[indices]<baseline[indices]))}
    return metrics,errors,baseline,diagnostic


def save_case(directory,fit,obs,power,positions,config,table,splits,axes,first,core,
              origin_xy,time_origin,start,window,reference,errors,baseline,diagnostic):
    directory.mkdir(parents=True,exist_ok=True)
    absolute_times=obs.times.numpy()+time_origin
    frequencies=obs.frequencies.numpy()
    fields=evaluate_field(fit.splauts,fit.proposals.relative_times,obs.frequencies).detach().numpy()
    fig,axs=plt.subplots(len(fields),1,figsize=(10,3*len(fields)),squeeze=False,constrained_layout=True)
    for k,ax in enumerate(axs[:,0]):
        emission=fit.proposals.relative_times.numpy()+float(fit.state.onsets[k])+time_origin
        ax.pcolormesh(emission,frequencies,fields[k].T,shading="auto",cmap="magma")
        ax.set(title=f"Candidate {k+1}: emitted spectral power shape",xlabel="Emission time (s)",ylabel="Hz")
    fig.savefig(directory/"sources.png",dpi=115)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    for ax,key in zip(axs,("positions","heldout_nmse")):
        if key=="heldout_nmse":
            points=ax.scatter(*table.xy[splits["test"]].T,c=errors[splits["test"]],vmin=0,vmax=2,s=20,cmap="viridis")
            fig.colorbar(points,ax=ax,label="Test-station NMSE (color capped at 2)")
        else:
            ax.scatter(*table.xy[splits["test"]].T,s=9,label=f"test ({len(splits['test'])})",alpha=.6)
        ax.scatter(*table.xy[splits["validation"]].T,s=20,marker="s",label="validation")
        ax.scatter(*table.xy[splits["fit"]].T,s=45,marker="^",color="orange",label="fit")
        xy=fit.state.positions.numpy()+origin_xy
        ax.scatter(*xy.T,s=170,marker="*",color="red",label="inferred candidates")
        for k,point in enumerate(xy):
            ax.annotate(str(k+1),point)
        ax.set(xlabel="East of M04 (m)",ylabel="North of M04 (m)",aspect="equal",title=key)
    axs[0].legend(fontsize=8)
    fig.savefig(directory/"map.png",dpi=120)
    plt.close(fig)
    fig,axs=plt.subplots(len(diagnostic),3,figsize=(13,2*len(diagnostic)),squeeze=False,constrained_layout=True)
    for row,(i,prediction) in zip(axs,diagnostic):
        values=[power[i].numpy(),prediction.numpy(),power[i].numpy()-prediction.numpy()]
        vmax=float(np.quantile(values[0][core.numpy()],.995))
        for ax,value,title in zip(row,values,("observed","predicted","residual")):
            ax.pcolormesh(absolute_times,frequencies,value.T/max(vmax,1e-12),shading="auto",
                          cmap="coolwarm" if title=="residual" else "magma",vmin=-1 if title=="residual" else 0,vmax=1)
            label="fit" if i in splits["fit"] else "TEST"
            ax.set(title=f"{table.ids[i]} {label}: {title}",xlim=(start,start+window),ylabel="Hz")
    for ax in axs[-1]:
        ax.set_xlabel("Seconds")
    fig.savefig(directory/"reconstruction.png",dpi=100)
    plt.close(fig)
    # Reference-channel mixture phase masks; same gain for every exported wave.
    z=axes["reference_z"][first:first+len(obs.times)]
    band=axes["band"]
    components=render_components(fit,positions[reference:reference+1],obs.times,obs.frequencies,config)[:,0].numpy()
    raw=np.abs(z)**2
    denominator=np.maximum(raw[:,band],components.sum(0)+axes["background"][reference])
    masks=components/np.maximum(denominator,1e-20)
    full=np.zeros_like(raw)
    full[:,band]=1
    mixture=audio_from_mask(z,full,500,500,50)
    waves=[]
    for mask in masks:
        full=np.zeros_like(raw)
        full[:,band]=mask
        waves.append(audio_from_mask(z,full,500,500,50))
    waves.append(mixture-np.sum(waves,axis=0))
    a=round((start-time_origin)*500)
    b=a+round(window*500)
    gain=.8/max(float(np.abs(mixture[a:b]).max()),1e-12)
    for name,wave in [("mixture",mixture),*[(f"component_{k+1}",v) for k,v in enumerate(waves[:-1])],("residual",waves[-1])]:
        sf.write(directory/f"{name}.wav",wave[a:b]*gain,500,subtype="FLOAT")
        sf.write(directory/f"{name}_4x.wav",wave[a:b]*gain,2000,subtype="FLOAT")
    torch.save({"config":asdict(config),"generator":fit.generator.state_dict(),"state":asdict(fit.state),
                "splauts":asdict(fit.splauts),"fit_observation":obs.power,"fit_stations":obs.stations,
                "times":obs.times,"frequencies":obs.frequencies,"relative_times":fit.proposals.relative_times,
                "coordinate_origin":torch.as_tensor(origin_xy),"time_origin":time_origin,
                "fit_indices":splits["fit"].tolist(),"time_mask":core,"objectives":fit.objectives,"best_step":fit.best_step},directory/"fit.pt")
    np.savez(directory/"station_errors.npz",nmse=errors,interpolation_nmse=baseline)


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    table,splits,data=prepare(args.audio,args.metadata or args.audio.with_suffix(".csv"),args.data,
                              args.radius,args.fit_stations,args.validation_stations,args.split_seed,args.low_hz,args.high_hz)
    if args.prepare_only:
        return
    args.output.mkdir(parents=True,exist_ok=True)
    power=np.load(args.data/"power.npy",mmap_mode="r")
    with np.load(args.data/"axes.npz") as handle:
        axes={key:handle[key] for key in handle.files}
    frequencies=torch.as_tensor(axes["frequencies"],dtype=torch.float32)
    area=float(np.ptp(table.xy,axis=0).max()+2*args.margin)
    origin_xy=(table.xy.min(0)+table.xy.max(0)-area)/2
    positions=torch.as_tensor(table.xy-origin_xy,dtype=torch.float32)
    config=Config(sample_rate=500,n_fft=500,hop_length=50,scene_duration=args.window,
                  call_duration=args.call_duration,sound_speed=330.,distance_offset=args.distance_offset,
                  area_size=area,splauts=args.splauts,power_spreading_exponent=args.spreading)
    corners=torch.tensor([[0.,0.],[0.,area],[area,0.],[area,area]])
    padding=float(np.ceil(torch.cdist(corners,positions).max().item()/330+.5))
    duration=np.floor(data["wav"]["duration_s"]*10)/10
    starts=window_starts(duration,args.window,args.hop) if args.starts is None else np.array(args.starts)
    if (starts<0).any() or (starts+args.window>duration+1e-6).any():
        raise ValueError("Requested windows are outside the recording")
    protocol={"data":str(args.data.resolve()),"data_signature":data["signature"],"config":asdict(config),
              "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "padding_s":padding,"search_origin_east_north_m":origin_xy.tolist(),
              "splits":data["splits"],"reference_station":data["reference_station"],
              "selection":"Proposals and fit objective score only core arrival frames; padded samples support source-field initialization. Checkpoints use fit objective only. K selected per window by validation mean NMSE; test excluded.",
              "caveats":[data["mapping"]["mapping_assumption"],
                         "No event identities, clean source recordings or location ground truth.",
                         "K and 5 s duration are capacity priors; overlapping windows repeat candidates.",
                         "330 m/s fixed speed, stationary point sources and power spreading are propagation assumptions.",
                         "Held-out channels estimate their own recording-wide stationary floor only; no held-out transient gain fitting.",
                         "Mixture-phase reference masks are candidate components, not proof of clean elephant separation."]}
    protocol_path=args.output/"protocol.json"
    if args.resume and protocol_path.exists():
        old=json.loads(protocol_path.read_text())
        for p in (old,protocol):
            p["arguments"].pop("resume",None)
            p["arguments"].pop("report_only",None)
        # JSON persists Config tuple fields as lists; compare their serialized form.
        if json.dumps(old,sort_keys=True) != json.dumps(protocol,sort_keys=True):
            raise ValueError("Resume settings differ from recorded protocol")
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    results,catalog=[],[]
    for wi,start in enumerate(starts):
        first=max(0,int(np.floor((start-padding)*10)))
        last=min(power.shape[1],int(np.ceil((start+args.window+padding)*10))+1)
        time_origin=float(axes["times"][first])
        times=torch.arange(last-first,dtype=torch.float32)/10
        local=torch.as_tensor(np.array(power[:,first:last]),dtype=torch.float32)
        core=(times>=start-time_origin-1e-5)&(times<start+args.window-time_origin-1e-5)
        obs=MixtureObservations(local[splits["fit"]],positions[splits["fit"]],times,frequencies)
        for count in args.events:
            name=f"window_{start:06.2f}_K{count}"
            directory=args.output/name
            if args.resume and (directory/"metrics.json").exists() and (directory/"fit.pt").exists():
                row=json.loads((directory/"metrics.json").read_text())
                results.append(row)
                catalog.extend(row["candidates"])
                print(f"Resume {name}",flush=True)
                continue
            print(f"[{wi+1}/{len(starts)}] {name}, fit={len(splits['fit'])} test={len(splits['test'])}",flush=True)
            tic=time.perf_counter()
            fit=fit_mixtures(obs,config,count,args.warmup_steps,args.steps,geometry_steps=0,
                              proposal_grid_size=args.grid_size,fit_seed=args.seed,
                              bandwidth_hz=args.high_hz-args.low_hz,log_every=0,time_mask=core)
            metrics,errors,baseline,diagnostic=evaluate(fit,local,positions,times,frequencies,config,splits,core)
            candidates=[]
            fields=evaluate_field(fit.splauts,fit.proposals.relative_times,frequencies).detach()
            for k in range(count):
                xy=fit.state.positions[k].numpy()+origin_xy
                energy=fields[k].sum(1)
                centroid=float((energy*fit.proposals.relative_times).sum()/energy.sum())
                candidate={"window_start_s":float(start),"K":count,"component":k+1,
                           "east_m":float(xy[0]),"north_m":float(xy[1]),
                           "lat":float(table.origin[0]+xy[1]/111194.9266),
                           "lon":float(table.origin[1]+xy[0]/(111194.9266*np.cos(np.deg2rad(table.origin[0])))),
                           "emission_centroid_s":time_origin+float(fit.state.onsets[k])+centroid}
                candidates.append(candidate)
            row={"window_start_s":float(start),"K":count,"directory":name,"metrics":metrics,
                 "best_step":fit.best_step,"seconds":time.perf_counter()-tic,"candidates":candidates}
            save_case(directory,fit,obs,local,positions,config,table,splits,axes,first,core,
                      origin_xy,time_origin,float(start),args.window,data["reference_index"],errors,baseline,diagnostic)
            (directory/"metrics.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
            results.append(row)
            catalog.extend(candidates)
            (args.output/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
            print(f"  NMSE fit {metrics['fit']['mean_nmse']:.3f}, validation {metrics['validation']['mean_nmse']:.3f}, test {metrics['test']['mean_nmse']:.3f}",flush=True)
    (args.output/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    selected=[min([r for r in results if r["window_start_s"]==start],key=lambda r:r["metrics"]["validation"]["mean_nmse"]) for start in starts]
    (args.output/"selected.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    with (args.output/"candidates.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)
    report(args.output,protocol,data,results,selected)


def report(output,protocol,data,results,selected):
    from shutil import copy2
    for name in ("overview.png","fit_station_spectrograms.png","data_protocol.json"):
        source=Path(protocol["data"])/name
        destination=output/name
        if source.resolve()!=destination.resolve():
            copy2(source,destination)
    selected_catalog=[c for row in selected for c in row["candidates"]]
    with (output/"selected_candidates.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(selected_catalog[0]))
        writer.writeheader()
        writer.writerows(selected_catalog)
    summary={}
    for key,group in [(f"K={k}",[r for r in results if r["K"]==k]) for k in sorted({r["K"] for r in results})]+[("validation-selected",selected)]:
        summary[key]={"windows":len(group),"median_fit_nmse":float(np.median([r["metrics"]["fit"]["mean_nmse"] for r in group])),
                      "median_validation_nmse":float(np.median([r["metrics"]["validation"]["mean_nmse"] for r in group])),
                      "median_test_nmse":float(np.median([r["metrics"]["test"]["mean_nmse"] for r in group])),
                      "mean_test_nmse":float(np.mean([r["metrics"]["test"]["mean_nmse"] for r in group])),
                      "median_test_station_median_nmse":float(np.median([r["metrics"]["test"]["median_nmse"] for r in group])),
                      "median_interpolation_test_nmse":float(np.median([r["metrics"]["test"]["interpolation_mean_nmse"] for r in group])),
                      "test_windows_below_zero":sum(r["metrics"]["test"]["mean_nmse"]<1 for r in group),
                      "test_windows_beating_interpolation":sum(r["metrics"]["test"]["mean_nmse"]<r["metrics"]["test"]["interpolation_mean_nmse"] for r in group)}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    findings=""
    if (output/"spreading_validation.json").exists():
        comparison=json.loads((output/"spreading_validation.json").read_text())
        findings+=(f'<p>Power distance exponent 1 was selected using validation data: median NMSE '
                   f'{comparison["surface_like"]["median_validation_nmse"]:.3f}, versus '
                   f'{comparison["spherical"]["median_validation_nmse"]:.3f} for exponent 2. '
                   '<a href="spreading_validation.json">Propagation comparison</a>.</p>')
    if (output/"injection.json").exists():
        injected=json.loads((output/"injection.json").read_text())
        findings+='<h2>Known overlapping calls injected into measured background</h2><p>This favorable +12 dB test has known K/duration and matched propagation. It is not native-elephant ground truth.</p><ul>'
        for k,row in enumerate(injected["sources"]):
            findings+=(f'<li>Call {k+1}: location error {row["position_error_m"]:.1f} m; '
                       f'reference SI-SDR {row["mixture_si_sdr_db"]:.1f} → {row["separated_si_sdr_db"]:.1f} dB.</li>')
        findings+=(f'</ul><p>Mean test-station NMSE: {injected["nmse"]["test"]["mean"]:.4f}; '
                   f'median station NMSE: {injected["nmse"]["test"]["median"]:.4f}.</p>'
                   '<img src="injection/evaluation.png"><p><a href="injection.json">Injection metrics</a> · '
                   '<a href="injection/mixture_4x.wav">Mixture 4×</a> · <a href="injection/separated_1_4x.wav">Call 1</a> · '
                   '<a href="injection/separated_2_4x.wav">Call 2</a></p>')
    if (output/"controls.json").exists():
        controls=json.loads((output/"controls.json").read_text())
        findings+='<h2>Native-data geometry and decoder-seed controls</h2><table><tr><th>Window start</th><th>Original test NMSE</th><th>Restart test NMSE</th><th>Shuffled geometry test NMSE</th><th>Restart location changes (m)</th></tr>'
        for row in controls["windows"]:
            findings+=(f'<tr><td>{row["window_start_s"]:g}</td><td>{row["original"]["test"]["mean_nmse"]:.3f}</td>'
                       f'<td>{row["controls"]["restart"]["mean_nmse"]["test"]:.3f}</td>'
                       f'<td>{row["controls"]["shuffled_geometry"]["mean_nmse"]["test"]:.3f}</td>'
                       f'<td>{", ".join(f"{v:.1f}" for v in row["controls"]["restart"]["matched_position_change_m"])}</td></tr>')
        findings+='</table><p>One shuffle and restart per case. Displacements measure sensitivity, not localization accuracy. <a href="controls.json">Full controls</a>.</p>'
    if (output/"refinement.json").exists():
        refined=json.loads((output/"refinement.json").read_text())
        findings+='<h2>Additional capacity and optimization check</h2><p>96 splauts, 400 warmup and 800 joint-fit steps, with a 17×17 proposal grid, on the same three harmonic windows. This exploratory check does not replace the primary experiment.</p><table><tr><th>Start</th><th>K</th><th>Primary validation</th><th>Refined validation</th><th>Primary test</th><th>Refined test</th></tr>'
        for row in refined["cases"]:
            findings+=(f'<tr><td>{row["window_start_s"]:g}</td><td>{row["K"]}</td>'
                       f'<td>{row["primary_validation_nmse"]:.3f}</td><td>{row["refined_validation_nmse"]:.3f}</td>'
                       f'<td>{row["primary_test_nmse"]:.3f}</td><td>{row["refined_test_nmse"]:.3f}</td></tr>')
        findings+='</table><p>More capacity and optimization do not consistently resolve native reconstruction. <a href="refinement.json">Refinement metrics</a>.</p>'
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for count in sorted({r["K"] for r in results}):
        group=[r for r in results if r["K"]==count]
        for ax,key in zip(axs,("fit","test")):
            ax.plot([r["window_start_s"] for r in group],[r["metrics"][key]["mean_nmse"] for r in group],".-",label=f"K={count}")
            ax.set(title=f"{key} stations",xlabel="Window start (s)",ylabel="Mean station-balanced power NMSE",yscale="log")
            ax.axhline(1,color="gray",linestyle="--")
    baseline=[r["metrics"]["test"]["interpolation_mean_nmse"] for r in selected]
    axs[1].plot([r["window_start_s"] for r in selected],baseline,"k:",label="Fit-only spatial interpolation")
    for ax in axs: ax.legend()
    fig.savefig(output/"summary.png",dpi=130)
    plt.close(fig)
    summary_rows="".join(f'<tr><td>{key}</td><td>{r["median_fit_nmse"]:.3f}</td><td>{r["median_test_nmse"]:.3f}</td><td>{r["median_interpolation_test_nmse"]:.3f}</td></tr>' for key,r in summary.items())
    selected_names={r["directory"] for r in selected}
    rows=""
    for r in results:
        d=r["directory"]
        waveform_link=f'<a href="tdoa_{r["window_start_s"]:06.2f}.png">waveform map</a>' if (output/f'tdoa_{r["window_start_s"]:06.2f}.png').exists() else ""
        players="".join(f'<audio controls preload="none" src="{d}/component_{k+1}_4x.wav"></audio>' for k in range(r["K"]))
        rows+=(f'<tr><td>{r["window_start_s"]:g}</td><td>{r["K"]}{" *" if d in selected_names else ""}</td>'
               f'<td>{r["metrics"]["fit"]["mean_nmse"]:.3f}</td><td>{r["metrics"]["validation"]["mean_nmse"]:.3f}</td><td>{r["metrics"]["test"]["mean_nmse"]:.3f}</td>'
               f'<td><a href="{d}/map.png">map</a> · <a href="{d}/sources.png">sources</a> · <a href="{d}/reconstruction.png">station predictions</a> {waveform_link}</td>'
               f'<td><audio controls preload="none" src="{d}/mixture_4x.wav"></audio></td><td>{players}</td></tr>')
    caveats="".join(f"<li>{html.escape(c)}</li>" for c in protocol["caveats"])
    mapping=data["mapping"]
    document=f'''<!doctype html><meta charset="utf-8"><title>Seismic elephant-rumble POC</title>
<style>body{{font:16px system-ui;margin:30px;max-width:1500px}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}audio{{width:200px}}</style>
<h1>Seismic neural-splaut rumble experiment</h1>
<p>500 Hz data; 330 m/s propagation; {len(data['splits']['fit'])} randomly sampled fitting stations,
{len(data['splits']['validation'])} validation stations and {len(data['splits']['test'])} final test stations within {data['signature']['radius']:g} m of M04.
{protocol['arguments']['window']:g}-second windows, {protocol['arguments']['hop']:g}-second stride,
{protocol['arguments']['low_hz']:g}–{protocol['arguments']['high_hz']:g} Hz band, and learned shared Gaussian splaut decoders.
Only the central arrival-time interval enters proposal scoring and fitting; padding supports propagation.</p>
<p>Metadata has {mapping['csv_rows']} tracks but WAV has {mapping['wav_channels']} channels.
Unavailable CSV track indices are excluded; available track-to-channel ordering is assumed.</p>
<table><tr><th>Model</th><th>Median fit NMSE</th><th>Median test NMSE</th><th>Median spatial-interpolation test NMSE</th></tr>{summary_rows}</table>
<p>NMSE 1 is zero excess-power prediction. Each window averages normalized error over test stations;
medians above are across windows. Asterisks below mark K chosen using validation error only.
The baseline interpolates powers from the three nearest fitting stations without fitting test gains.
No source positions or clean elephant signals are available to score actual localization/separation accuracy.
Native held-out reconstruction remains weak in this first pass; source candidates are not reliable elephant locations.</p>
{findings}
<img src="summary.png"><details><summary>Grid and input spectrograms</summary>
<img src="overview.png"><img src="fit_station_spectrograms.png"></details>
<p>Audio is seismic vibration at fitting station {data['reference_station']}, played 4× faster.
All components/residual share their mixture gain. Original-rate WAVs and tensor checkpoints are in each case folder.</p>
<table><tr><th>Start (s)</th><th>K</th><th>Fit</th><th>Validation</th><th>Test</th><th>Diagnostics</th><th>Mixture 4×</th><th>Components 4×</th></tr>{rows}</table>
<h2>Assumptions and limitations</h2><ul>{caveats}</ul>
<p><a href="protocol.json">Protocol</a> · <a href="data_protocol.json">Station mapping and split</a> · <a href="results.json">All metrics</a> · <a href="selected.json">Validation-selected fits</a> · <a href="selected_candidates.csv">Selected candidate catalog</a> · <a href="candidates.csv">All candidate capacities</a></p>'''
    (output/"report.html").write_text(document,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio",type=Path,default=Path("real_world/source_data/elephant rumble/multi_seis_20230308_104500.wav"))
    p.add_argument("--metadata",type=Path)
    p.add_argument("--data",type=Path,default=Path("outputs/seismic_data"))
    p.add_argument("--output",type=Path,default=Path("outputs/seismic_rumbles"))
    p.add_argument("--radius",type=float,default=500.)
    p.add_argument("--fit-stations",type=int,default=16)
    p.add_argument("--validation-stations",type=int,default=16)
    p.add_argument("--split-seed",type=int,default=2026)
    p.add_argument("--window",type=float,default=8.)
    p.add_argument("--hop",type=float,default=4.)
    p.add_argument("--call-duration",type=float,default=5.)
    p.add_argument("--low-hz",type=float,default=8.)
    p.add_argument("--high-hz",type=float,default=80.)
    p.add_argument("--margin",type=float,default=150.)
    p.add_argument("--distance-offset",type=float,default=25.)
    p.add_argument("--spreading",type=float,default=1.,help="Power distance exponent: 1 surface-like; 2 spherical")
    p.add_argument("--events",type=int,nargs="+",choices=(1,2,3,4,5),default=[1,2])
    p.add_argument("--starts",type=float,nargs="+")
    p.add_argument("--splauts",type=int,default=64)
    p.add_argument("--warmup-steps",type=int,default=100)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--grid-size",type=int,default=15)
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--seed",type=int,default=101)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--report-only",action="store_true")
    args=p.parse_args()
    if args.report_only:
        protocol=json.loads((args.output/"protocol.json").read_text())
        data=json.loads((Path(protocol["data"])/"data_protocol.json").read_text())
        report(args.output,protocol,data,json.loads((args.output/"results.json").read_text()),json.loads((args.output/"selected.json").read_text()))
        return
    if (min(args.window,args.hop,args.call_duration,args.margin,args.distance_offset,args.spreading,args.threads,args.splauts)<=0
            or args.hop>args.window or args.call_duration>=args.window or args.grid_size<2
            or min(args.steps,args.warmup_steps)<0 or not 0<args.low_hz<args.high_hz<250):
        p.error("Invalid seismic analysis settings")
    run(args)


if __name__=="__main__":
    main()
