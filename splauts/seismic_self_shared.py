"""Self/shared neural splauts on native seismic calls, with unseen stations."""

import argparse
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
from .model.events import EventState
from .model.renderer import render
from .real_data import stft_power, audio_from_mask
from .seismic_data import select_stations
from .seismic_experiments import station_nmse, interpolate_stations
from .self_noise_experiments import location_clusters
from .training.mixtures import MixtureObservations
from .training.self_shared import SelfSharedLoss, fit_self_shared, received_components, decomposition_loss


def correlation(prediction, target):
    """Per-station temporal correlation after centering each frequency bin."""
    a = prediction - prediction.mean(1, keepdim=True)
    b = target - target.mean(1, keepdim=True)
    return (a*b).sum((1,2)) / (a.square().sum((1,2))*b.square().sum((1,2))).clamp_min(1e-20).sqrt()


def assess(prediction, target, baseline, splits):
    p = prediction - prediction.mean(1, keepdim=True)
    y = target - target.mean(1, keepdim=True)
    b = baseline - baseline.mean(1, keepdim=True)
    arrays = {
        "raw_nmse": station_nmse(prediction, target),
        "dynamic_nmse": station_nmse(p, y),
        "dynamic_correlation": correlation(prediction, target),
        "interpolation_raw_nmse": station_nmse(baseline, target),
        "interpolation_dynamic_nmse": station_nmse(b, y),
        "interpolation_dynamic_correlation": correlation(baseline, target),
        "shared_observed_energy_fraction": torch.minimum(prediction, target).sum((1,2))/target.sum((1,2)).clamp_min(1e-20),
        "shared_overprediction_fraction": (prediction-target).clamp_min(0).sum((1,2))/target.sum((1,2)).clamp_min(1e-20),
    }
    arrays = {k: v.detach().numpy() for k,v in arrays.items()}
    metrics = {s: {k: {"mean": float(v[ii].mean()), "median": float(np.median(v[ii]))}
                   for k,v in arrays.items()} for s,ii in splits.items()}
    return metrics, arrays


def plots(directory, fit, prediction, power, times, frequencies, core, table, splits, origin, start):
    centers, groups = location_clusters(fit.state.positions, fit.gates)
    fig, ax = plt.subplots(figsize=(8,7), constrained_layout=True)
    for key, marker, size in [("test",".",14),("validation","s",18),("fit","^",55)]:
        ax.scatter(*table.xy[splits[key]].T, marker=marker, s=size, label=key)
    xy = fit.state.positions.numpy()+origin
    for k, point in enumerate(xy):
        active = fit.gates[k]>=.03
        ax.scatter(*point, marker="*" if active else "x", s=170 if active else 40, color="red" if active else "grey",
                   label=("active group" if active else "inactive proposal") if k==0 else None)
        ax.annotate(f"{k+1} ({float(fit.gates[k]):.2f})",point)
    ax.scatter(0,0,marker="+",color="black",label="M04")
    ax.set(xlabel="East of M04 (m)",ylabel="North of M04 (m)",title=f"{start:g}–{start+8:g} s; {len(groups)} active sites",aspect="equal")
    ax.legend()
    fig.savefig(directory/"map.png",dpi=120); plt.close(fig)
    # Pick diagnostics from geometry, including the established M04-nearest fit
    # reference, rather than selecting stations by reconstruction quality.
    near=lambda indices: indices[np.argsort(np.linalg.norm(table.xy[indices],axis=1))]
    chosen=np.r_[near(splits["fit"])[:2],near(splits["test"])[:2],splits["test"][:1]]
    fig, axs=plt.subplots(len(chosen),3,figsize=(14,2.4*len(chosen)),constrained_layout=True)
    for row,i in zip(axs,chosen):
        y, p=power[i].numpy(),prediction[i].numpy()
        vmax=max(float(np.quantile(y[core],.995)),1e-12)
        for ax,value,title in zip(row,[y,p,np.maximum(y-p,0)],["observed","shared prediction","self residual"]):
            ax.pcolormesh(times,frequencies,10*np.log10(np.maximum(value.T/vmax,1e-5)),vmin=-40,vmax=0,cmap="magma",shading="auto")
            ax.set(xlim=(start,start+8),ylabel="Hz",title=f"{table.ids[i]} {'FIT' if i in splits['fit'] else 'TEST'}: {title}")
    for ax in axs[-1]: ax.set_xlabel("Seconds from recording start")
    fig.savefig(directory/"decomposition.png",dpi=110); plt.close(fig)
    fields=evaluate_field(fit.splauts,fit.proposals.relative_times,torch.as_tensor(frequencies)).detach().numpy()
    fig,axs=plt.subplots(len(fields),1,figsize=(11,2.2*len(fields)),squeeze=False,constrained_layout=True)
    offset=float(times[0])
    for k,ax in enumerate(axs[:,0]):
        t=fit.proposals.relative_times.numpy()+float(fit.state.onsets[k])+offset
        ax.pcolormesh(t,frequencies,fields[k].T,shading="auto",cmap="magma")
        ax.set(ylabel="Hz",title=f"Group {k+1}: gate {float(fit.gates[k]):.3f}, support {float(fit.support[k]):.1f}")
    axs[-1,0].set_xlabel("Emission time (s)")
    fig.savefig(directory/"sources.png",dpi=110); plt.close(fig)


def export_audio(directory, fit, stations, times, frequencies, config, reference_z, band, first, core, reference):
    components=[]
    with torch.no_grad():
        for k in range(len(fit.gates)):
            one=Splauts(**{key:getattr(fit.splauts,key)[k:k+1] for key in vars(fit.splauts)})
            components.append(render(one,fit.state.positions[k:k+1],fit.state.onsets[k:k+1],fit.state.amplitudes[k:k+1],
                                     stations[reference:reference+1],times,frequencies,config)[0].numpy())
    components=np.array(components)
    z=reference_z[first:first+len(times)]
    raw=np.abs(z)**2
    denominator=np.maximum(raw[:,band],components.sum(0)).clip(1e-20)
    full=np.zeros_like(raw); full[:,band]=1
    mixture=audio_from_mask(z,full,500,500,50)
    waves=[]
    for component in components:
        full=np.zeros_like(raw); full[:,band]=component/denominator
        waves.append(audio_from_mask(z,full,500,500,50))
    a=int(torch.nonzero(core)[0])*50; b=a+4000
    gain=.8/max(float(np.abs(mixture[a:b]).max()),1e-12)
    named={"mixture":mixture,"shared":np.sum(waves,axis=0),"self":mixture-np.sum(waves,axis=0)}
    named.update({f"group_{k+1}":v for k,v in enumerate(waves)})
    for name,wave in named.items():
        for factor in (1,4):
            sf.write(directory/f"{name}{'_4x' if factor==4 else ''}.wav",wave[a:b]*gain,500*factor,subtype="FLOAT")


def report(output, protocol):
    rows=[json.loads(p.read_text()) for p in sorted(output.glob("window_*/metrics.json"))]
    (output/"results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    native=[r for r in rows if r["control"]=="native"]
    summary={"native_windows":len(native),"active_sites_by_window":[r["active_sites"] for r in native]}
    for key in ("raw_nmse","dynamic_nmse","dynamic_correlation","shared_observed_energy_fraction","interpolation_raw_nmse","interpolation_dynamic_nmse"):
        summary["mean_window_test_"+key]=float(np.mean([r["metrics"]["test"][key]["mean"] for r in native])) if native else None
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    content='''<!doctype html><meta charset="utf-8"><title>Native seismic self/shared splauts: 30–50 s</title>
<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:15px;line-height:1.5}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}audio{max-width:100%}</style>
<h1>Native seismic calls: self/shared splauts, 30–50 seconds</h1>
<p>Four overlapping 8 s arrival windows; 16 fit, 16 validation and 181 evaluation-only test stations within 500 m of M04. Full BANDLABEL Hz power spectra, 500 Hz sampling, 1 s Hann windows / 0.1 s hops. Propagation fixed at 330 m/s, power exponent 1 and 25 m offset. Five potential groups, 48 Gaussians per group, 5 s duration prior. Every fit begins with zero shared energy and the full spectrogram in self. Loss settings transfer from the synthetic study without real-data tuning.</p>
<p>Only core arrival frames drive proposals, normalization, coherence and loss; 5 s padding provides propagation context. Checkpoints minimize the same fixed fit-only objective, including all-self. Test observations never set amplitudes, source fields or gates. Validation is reported, with no hyperparameter selection in this run. Active means gate ≥0.03; nearby active groups within 15 m form a reported site, not a known animal count. Windows repeat events.</p>
<h2>Prediction at unseen stations</h2><p>Scores below use <b>shared prediction alone</b>. Adding a test station’s self residual is not evidence of separation. Raw NMSE = 1 for zero prediction. Dynamic NMSE and correlation center each frequency bin over the core, removing stationary spectra for evaluation only; dynamic NMSE = 1 for a constant prediction. IDW interpolates the three nearest fitting spectrograms with no held-out gain fitting. These metrics use all 8–80 Hz bins, without selecting call frequencies from test data.</p>
<table><tr><th>Window / run</th><th>Sites</th><th>Self fraction (fit)</th><th>Test raw NMSE</th><th>Test dynamic NMSE</th><th>Test correlation</th><th>IDW raw / dynamic NMSE</th><th>Validation raw NMSE</th></tr>'''
    for r in rows:
        t=r["metrics"]["test"]; v=r["metrics"]["validation"]
        content+=f'<tr><td><a href="{r["directory"]}/metrics.json">{r["start"]:g}–{r["start"]+8:g} / {r["control"]}</a></td><td>{r["active_sites"]}</td><td>{r["self_fraction"]:.3f}</td><td>{t["raw_nmse"]["mean"]:.3f}</td><td>{t["dynamic_nmse"]["mean"]:.3f}</td><td>{t["dynamic_correlation"]["mean"]:.3f}</td><td>{t["interpolation_raw_nmse"]["mean"]:.3f} / {t["interpolation_dynamic_nmse"]["mean"]:.3f}</td><td>{v["raw_nmse"]["mean"]:.3f}</td></tr>'
    content+='</table><p>Means are station-balanced; JSON also contains medians, per-station arrays, captured energy and overprediction. Geometry shuffle permutes fitting coordinates while leaving test coordinates correct; time shuffle independently circular-shifts fitting spectrograms. Restart changes only initialization. Controls use the 38–46 s window. This is an exploratory first pass, not a preregistered blind benchmark.</p>'
    if native and all(c["gate"]==0 for r in native for c in r["candidates"]):
        content+='<h2>Finding: the sparse model remains all-self</h2><p>No shared event passes the fixed objective in these windows. Shared predictions are zero, so raw and dynamic NMSE are 1 and correlation is 0. This run does not separate or localize the known calls. All-self is a failure to extract them, not evidence that calls are absent.</p>'
    elif native:
        content+='<h2>Finding: candidate shared energy</h2><p>The model transfers some energy out of self. Assess held-out dynamic prediction and the controls before interpreting any active site as a source. There is no measured native localization accuracy.</p>'
    comparisons=[]
    for label,other in [("Broad band 8–80 Hz, original loss","seismic_self_shared_wide"),("8–35 Hz, spatial penalty removed (diagnostic)","seismic_self_shared_no_sparse")]:
        candidate=output.parent/other/"window_038.0_native"/"metrics.json"
        if candidate.exists():
            r=json.loads(candidate.read_text()); t=r["metrics"]["test"]
            comparisons.append(f'<tr><td><a href="../{other}/report.html">{label}</a></td><td>{r["active_sites"]}</td><td>{t["raw_nmse"]["mean"]:.3f}</td><td>{t["dynamic_nmse"]["mean"]:.3f}</td><td>{t["dynamic_correlation"]["mean"]:.3f}</td><td>{r["metrics"]["validation"]["raw_nmse"]["mean"]:.3f}</td></tr>')
    if comparisons:
        content+='<h2>38–46 s sensitivity checks</h2><p>The initial short broad-band pilot proposed mostly 45–75 Hz energy and remained all-self. The main run restricts the band to 8–35 Hz to focus on rumbles. The no-sparsity diagnostic tests whether the spatial cost blocks activation; it is not the requested sparse model, and none of these alternatives is selected by test scores.</p><table><tr><th>Run</th><th>Sites</th><th>Test raw NMSE</th><th>Test dynamic NMSE</th><th>Test correlation</th><th>Validation raw NMSE</th></tr>'+''.join(comparisons)+'</table>'
    extended=output.parent/"seismic_self_shared_explore"/"results.json"
    if extended.exists():
        content+='<h2>Longer spatial exploration</h2><p><a href="../seismic_self_shared_explore/report.html">Extended-continuation trial and controls</a>: zero spatial optimization penalty for the first 200 steps, then a ramp to the full 0.08 penalty by step 500. Every checkpoint, including the initial all-self model, is selected using the same full final loss. This addresses an activation barrier without selecting an unpenalized final model.</p><table><tr><th>Run (38–46 s)</th><th>Sites</th><th>Test raw NMSE</th><th>Test dynamic NMSE</th><th>Test correlation</th></tr>'
        for r in json.loads(extended.read_text()):
            t=r["metrics"]["test"]
            content+=f'<tr><td>{r["control"]}</td><td>{r["active_sites"]}</td><td>{t["raw_nmse"]["mean"]:.3f}</td><td>{t["dynamic_nmse"]["mean"]:.3f}</td><td>{t["dynamic_correlation"]["mean"]:.3f}</td></tr>'
        content+='</table>'
        trial=[r for r in json.loads(extended.read_text()) if r["control"]=="native"]
        if trial and trial[0]["metrics"]["test"]["dynamic_nmse"]["mean"]>=.99 and trial[0]["metrics"]["test"]["dynamic_correlation"]["mean"]<.1:
            content+='<p><b>The continuation trial does not establish rumble extraction.</b> It can lower the sparse fit objective and raw held-out error, but it provides essentially no improvement in time-varying held-out energy. Its active positions should not be treated as elephant locations. The native no-sparsity and extended-continuation trials cover only 38–46 s, while the main transferred-loss run covers all of 30–50 s.</p>'
            content+='<p>The panel below includes the established reference 13_05 and nearby test stations, chosen by geometry. At 13_05 the visible low-frequency bursts around 42 and 45 s remain in self; much of the predicted shared power at other stations is spectrally broad and slowly varying.</p><img src="../seismic_self_shared_explore/window_038.0_native/decomposition.png">'
            content+='<p>Continuation-trial reference audio, 4× playback: <a href="../seismic_self_shared_explore/window_038.0_native/mixture_4x.wav">mixture</a> · <a href="../seismic_self_shared_explore/window_038.0_native/shared_4x.wav">shared</a> · <a href="../seismic_self_shared_explore/window_038.0_native/self_4x.wav">self</a>.</p>'
    for r in native:
        d=r["directory"]
        content+=f'<h2>{r["start"]:g}–{r["start"]+8:g} seconds</h2><p>Best step {r["best_step"]}; objective {r["objective"]:.4f} vs all-self 1.0. Active groups: '
        content+=html.escape(json.dumps([c for c in r["candidates"] if c["active"]]))+'</p>'
        content+=f'<img src="{d}/map.png"><img src="{d}/decomposition.png"><details><summary>All proposed source shapes</summary><img src="{d}/sources.png"></details>'
        for name in ("mixture","shared","self"):
            content+=f'<p>{name}, 4× playback: <audio controls src="{d}/{name}_4x.wav"></audio> <a href="{d}/{name}.wav">native 500 Hz WAV</a></p>'
        content+='<p>Mixture-phase masks at fitting reference 13_05; identical gain for all outputs. Playback is vibration made audible, not a clean acoustic recording. Individual group WAVs are in the window directory.</p>'
    content+='''<h2>Interpretation limits</h2><p>There are no clean native sources or surveyed elephant locations. Captured shared energy can include other coherent sources; candidates and masks do not establish event identity, source count or localization accuracy. Station response variation, scattering, coherent background and an imperfect point-source propagation law can all limit prediction. The CSV has 1,153 tracks versus 1,024 WAV channels: available one-based TrackIndex values are assumed to map directly to WAV channels; export ordering is unverified.</p><p><a href="protocol.json">Protocol</a> · <a href="results.json">Results</a> · <a href="summary.json">Summary</a></p>'''
    content+='<h2>Next experiment suggested by the failure</h2><p>Keep a stationary per-station noise floor in self and reward transfer of time-varying excess power. Propose a group using only part of the fitting array, then require it to predict the remaining fitting stations before accepting a location. Keep final evaluation stations separate. This directly tests predictive sharing rather than allowing a flexible source field to earn energy credit for broad background. Verify the TrackIndex/channel mapping before interpreting any native location.</p>'
    if (output/"audit.json").exists():
        content+='<p><a href="audit.json">Saved-checkpoint audit</a>: rerendered physical amplitudes, reproduced the fixed core objective and minimum-objective selection, verified all-self initialization, finite station metrics and shared+self audio sums.</p>'
    content=content.replace("BANDLABEL",f"{protocol['band_hz'][0]:g}–{protocol['band_hz'][1]:g}").replace("all 8–80 Hz bins","all fitted frequency bins")
    if protocol["weights"]["spatial_sparsity"]!=.08:
        content=content.replace("Loss settings transfer from the synthetic study without real-data tuning.",f"Diagnostic spatial penalty: {protocol['weights']['spatial_sparsity']}; other loss settings transfer from the synthetic study.")
    if protocol.get("spatial_exploration_steps",0):
        content=content.replace("Loss settings transfer from the synthetic study without real-data tuning.","Final loss weights transfer unchanged from the synthetic study; this trial extends spatial exploration to 200 steps before ramping the spatial penalty to its full value by step 500.")
    content=content.replace("Four overlapping 8 s arrival windows",f"{len(native)} completed 8 s arrival windows")
    (output/"report.html").write_text(content,encoding="utf-8")


def audit(output):
    """Rerender saved physical amplitudes and verify conservative selection."""
    torch.set_num_threads(1); torch.set_flush_denormal(True)
    results=[]
    for path in sorted(output.glob("window_*/fit.pt")):
        ck=torch.load(path,map_location="cpu",weights_only=False)
        config=Config(**ck["config"]); weights=SelfSharedLoss(**ck["weights"])
        state=EventState(**ck["state"]); splauts=Splauts(**ck["splauts"])
        obs=MixtureObservations(ck["fit_power"],ck["fit_stations"],ck["times"],ck["frequencies"])
        with torch.no_grad():
            physical=render(splauts,state.positions,state.onsets,state.amplitudes,obs.stations,obs.times,obs.frequencies,config)
            torch.testing.assert_close(physical,ck["shared"],rtol=3e-5,atol=1e-6)
            components,support,amplitudes=received_components(splauts,state,obs,config,ck["gates"],weights,ck["time_mask"])
            torch.testing.assert_close(amplitudes,state.amplitudes,rtol=3e-5,atol=1e-6)
            torch.testing.assert_close(support,ck["support"])
            loss,local,_=decomposition_loss(components,obs.power,ck["gates"],support,weights,state.positions,ck["time_mask"])
            torch.testing.assert_close(local,ck["self_power"],rtol=3e-5,atol=1e-6)
        history=ck["history"]
        assert history[0]["gate_mass"]==[0.]*len(ck["gates"])
        assert history[0]["self_energy"]==1. and history[0]["reconstruction"]==0.
        best=min(h["objective"] for h in history)
        assert abs(float(loss)-best)<1e-5, (path,float(loss),best)
        assert abs(history[ck["best_step"]]["objective"]-best)<1e-7
        if (path.parent/"mixture.wav").exists():
            waves=[sf.read(path.parent/f"{n}.wav")[0] for n in ("mixture","shared","self")]
            assert all(len(w)==4000 for w in waves)
            np.testing.assert_allclose(waves[1]+waves[2],waves[0],rtol=1e-5,atol=1e-7)
        with np.load(path.parent/"station_metrics.npz") as arrays:
            assert all(np.isfinite(arrays[k]).all() and arrays[k].shape==(213,) for k in arrays.files)
        results.append({"directory":path.parent.name,"objective":float(loss),"best_step":ck["best_step"],"passed":True})
    if not results: raise ValueError("No checkpoints to audit")
    (output/"audit.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"Audited {len(results)} checkpoints: physical rerender, core objective, all-self start, selection, station metrics and audio sums",flush=True)


def run(args):
    torch.set_num_threads(args.threads); torch.set_flush_denormal(True)
    args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads((args.data/"data_protocol.json").read_text())
    signature=data["signature"]
    audio_stat=Path(signature["audio"]).stat()
    if (audio_stat.st_size!=signature["size"] or audio_stat.st_mtime_ns!=signature["mtime_ns"]
            or Path(signature["metadata"]).stat().st_mtime_ns!=signature["metadata_mtime_ns"]):
        raise ValueError("Source data changed since cache preparation")
    source=Path(data["signature"]["metadata"])
    table,splits,mapping=select_stations(source,data["wav"]["channels"])
    if {k:v.tolist() for k,v in splits.items()} != data["splits"]:
        raise ValueError("Cached split mismatch")
    if table.ids!=[s["id"] for s in data["stations"]] or table.tracks.tolist()!=[s["track_index"]-1 for s in data["stations"]]:
        raise ValueError("Cached station ordering mismatch")
    with np.load(args.data/"axes.npz") as handle: axes={k:handle[k] for k in handle.files}
    band=(np.fft.rfftfreq(500,1/500)>=args.low_hz)&(np.fft.rfftfreq(500,1/500)<=args.high_hz)
    audio=np.load(args.data/"audio.npy",mmap_mode="r")
    # Recompute exact full power, including bins that the older floor subtraction clipped.
    powers=[]
    for a in range(0,len(audio),16):
        z,times,f=stft_power(audio[a:a+16],500,500,50)
        powers.append((np.abs(z[...,band])**2/data["power_scale"]).astype(np.float32))
    full_power=np.concatenate(powers)
    area=float(np.ptp(table.xy,axis=0).max()+300)
    origin=(table.xy.min(0)+table.xy.max(0)-area)/2
    stations=torch.as_tensor(table.xy-origin,dtype=torch.float32)
    frequencies=torch.as_tensor(f[band],dtype=torch.float32)
    config=Config(sample_rate=500,n_fft=500,hop_length=50,scene_duration=8.,call_duration=5.,
                  area_size=area,sound_speed=330.,distance_offset=25.,splauts=48,power_spreading_exponent=1.)
    weights=SelfSharedLoss(spatial_sparsity=args.spatial_penalty)
    protocol={"config":asdict(config),"weights":asdict(weights),"splits":data["splits"],"mapping":mapping,
              "power_scale":data["power_scale"],"preprocessing":"Full original power; no floor subtraction; common fit-only numerical scale",
              "source_signature":data["signature"],"search_origin_m":origin.tolist(),"padding_s":5,
              "steps":args.steps,"warmup_steps":args.warmup_steps,"seed":101,"capacity":5,"grid_size":15,"band_hz":[args.low_hz,args.high_hz],
              "spatial_exploration_steps":args.exploration_steps,
              "native_windows_s":[30,34,38,42],"control_window_s":38,
              "selection":"Fixed final fit objective; no real-data hyperparameter selection; validation/test excluded from source fitting"}
    pp=args.output/"protocol.json"
    if pp.exists():
        old=json.loads(pp.read_text()); old.setdefault("spatial_exploration_steps",0)
        if old!=json.loads(json.dumps(protocol)):
            raise ValueError("Protocol differs; choose a new output directory")
    pp.write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    for start in args.starts:
        first=round((start-5)*10); last=round((start+13)*10)+1
        power=torch.as_tensor(full_power[:,first:last].copy())
        local_times=torch.arange(last-first,dtype=torch.float32)/10
        core=(local_times>=5)&(local_times<13)
        for control in args.controls:
            name=f"window_{start:05.1f}_{control}"; directory=args.output/name
            required=["metrics.json","fit.pt","station_metrics.npz"]
            if control=="native":
                required += ["map.png","decomposition.png","sources.png","mixture.wav","shared.wav","self.wav",
                             "mixture_4x.wav","shared_4x.wav","self_4x.wav"]
            if args.resume and all((directory/name).exists() for name in required):
                print(f"Resume {name}",flush=True); continue
            directory.mkdir(parents=True,exist_ok=True)
            fit_power=power[splits["fit"]].clone(); fit_xy=stations[splits["fit"]].clone()
            rng=np.random.default_rng(2038)
            if control=="geometry_shuffle": fit_xy=fit_xy[rng.permutation(len(fit_xy))]
            if control=="time_shuffle":
                for m,shift in enumerate(rng.integers(30,len(local_times)-30,len(fit_power))):
                    fit_power[m]=torch.roll(fit_power[m],int(shift),dims=0)
            obs=MixtureObservations(fit_power,fit_xy,local_times,frequencies)
            print(f"START {name}",flush=True); tic=time.perf_counter()
            fit=fit_self_shared(obs,config,capacity=5,weights=weights,warmup_steps=args.warmup_steps,
                                steps=args.steps,geometry_steps=75,grid_size=15,
                                seed=202 if control=="restart" else 101,time_mask=core,log_every=100,
                                spatial_exploration_steps=args.exploration_steps)
            predictions=[]
            with torch.no_grad():
                for a in range(0,len(power),16):
                    predictions.append(render(fit.splauts,fit.state.positions,fit.state.onsets,fit.state.amplitudes,
                                              stations[a:a+16],local_times,frequencies,config))
            prediction=torch.cat(predictions)
            baseline=interpolate_stations(power[splits["fit"]],stations[splits["fit"]],stations)
            metrics,arrays=assess(prediction[:,core],power[:,core],baseline[:,core],splits)
            centers,groups=location_clusters(fit.state.positions,fit.gates)
            fields=evaluate_field(fit.splauts,fit.proposals.relative_times,frequencies).detach()
            candidates=[]
            for k in range(5):
                energy=fields[k].sum(-1); centroid=float((energy*fit.proposals.relative_times).sum()/energy.sum().clamp_min(1e-20))
                xy=fit.state.positions[k].numpy()+origin
                candidates.append({"group":k+1,"active":bool(fit.gates[k]>=.03),"gate":float(fit.gates[k]),
                                   "support":float(fit.support[k]),"east_m":float(xy[0]),"north_m":float(xy[1]),
                                   "emission_centroid_s":start-5+float(fit.state.onsets[k])+centroid})
            best=fit.history[fit.best_step]
            row={"start":start,"control":control,"directory":name,"seconds":time.perf_counter()-tic,
                 "best_step":fit.best_step,"last_step":fit.history[-1]["step"],"objective":best["objective"],"self_fraction":best["self_energy"],
                 "active_sites":len(groups),"site_centers_east_north_m":(centers.numpy()+origin).tolist(),
                 "site_groups":groups,"metrics":metrics,"candidates":candidates}
            torch.save({"config":asdict(config),"weights":asdict(weights),"state":asdict(fit.state),
                        "splauts":asdict(fit.splauts),"generator":fit.generator.state_dict(),
                        "gates":fit.gates,"support":fit.support,"fit_power":obs.power,"fit_stations":obs.stations,
                        "times":local_times,"frequencies":frequencies,"time_mask":core,"history":fit.history,
                        "shared":fit.shared,"self_power":fit.self_power,"coordinate_origin":origin,"time_origin":start-5,
                        "best_step":fit.best_step},directory/"fit.pt")
            np.savez(directory/"station_metrics.npz",**arrays)
            if control=="native":
                plots(directory,fit,prediction,power,times[first:last],frequencies.numpy(),core.numpy(),table,splits,origin,start)
                export_audio(directory,fit,stations,local_times,frequencies,config,axes["reference_z"],band,first,core,data["reference_index"])
            temp=directory/"metrics.tmp"
            temp.write_text(json.dumps(row,indent=2),encoding="utf-8")
            temp.replace(directory/"metrics.json")
            report(args.output,protocol)
            print(f"DONE {name}: sites={len(groups)}, test raw={metrics['test']['raw_nmse']['mean']:.3f}, dynamic={metrics['test']['dynamic_nmse']['mean']:.3f}, corr={metrics['test']['dynamic_correlation']['mean']:.3f}",flush=True)
    report(args.output,protocol)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("outputs/seismic_data"))
    p.add_argument("--output",type=Path,default=Path("outputs/seismic_self_shared_low"))
    p.add_argument("--starts",type=float,nargs="+",default=[30,34,38,42])
    p.add_argument("--controls",nargs="+",choices=["native","restart","geometry_shuffle","time_shuffle"],default=["native"])
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--warmup-steps",type=int,default=200)
    p.add_argument("--low-hz",type=float,default=8.)
    p.add_argument("--high-hz",type=float,default=35.)
    p.add_argument("--spatial-penalty",type=float,default=.08)
    p.add_argument("--exploration-steps",type=int,default=0)
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--report-only",action="store_true")
    args=p.parse_args()
    if args.audit_only:
        audit(args.output); return
    if args.report_only:
        report(args.output,json.loads((args.output/"protocol.json").read_text())); return
    if min(args.steps,args.warmup_steps,args.spatial_penalty,args.exploration_steps)<0 or args.threads<1 or not 8<=args.low_hz<args.high_hz<=80 or any(s<30 or s+8>50 for s in args.starts):
        p.error("Use nonnegative steps, positive threads, and core windows within 30–50 s")
    run(args)


if __name__=="__main__": main()
