"""Moo calls and a paired spectral-absorption ablation with unseen-station prediction.

Run: python -m splauts.attenuation_experiments --prior outputs/moo_attenuation/prior/prior.pt
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import time
from types import SimpleNamespace
import wave

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .mixture_experiments import _save_sweep, _write_json, evaluate_saved_sources, record_mixture_fit
from .model.renderer import render
from .model.generator import Splauts
from .pretrained_experiments import prepare_prior
from .propagation import absorption_transmission
from .simulator.calls import CallSpec, instantaneous_frequency, power_spectrogram, synthesize_call
from .simulator.scene import random_scene, simulate_scene
from .training.frozen import fit_frozen_prior
from .training.mixtures import MixtureObservations, mixture_losses
from .training.pretrain import decoder_fingerprint
from .visualization.plots import _save, plot_stations


CONDITIONS = ("off", "modeled", "ignored")
HOLDOUT_STATIONS = torch.tensor([[60., 90.], [240., 105.], [110., 245.], [225., 235.]])


def _emission_fingerprint(scene) -> str:
    spec = [{"position": e.position.tolist(), "onset": e.onset, "amplitude": e.amplitude,
             "call": asdict(e.call)} for e in scene.events]
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def prediction_audit(scene, result, observed_config: Config, model_config: Config, output: Path | None = None) -> dict:
    """Held-out station waveforms are created only after the fit has been selected."""
    if torch.cdist(scene.stations, HOLDOUT_STATIONS).min() < 1.:
        raise ValueError("Prediction stations must be separate from fitting stations")
    heldout = simulate_scene(scene.events, HOLDOUT_STATIONS, observed_config, seed=observed_config.seed+100003)
    clean = simulate_scene(scene.events, HOLDOUT_STATIONS, replace(observed_config, noise_std=0.))
    state = result.state
    prediction = render(result.splauts, state.positions, state.onsets, state.amplitudes,
                        HOLDOUT_STATIONS, heldout.times, heldout.frequencies, model_config)
    noisy_error, _ = mixture_losses(prediction, heldout.power)
    clean_error, _ = mixture_losses(prediction, clean.power)
    if output is not None:
        torch.save({"stations": HOLDOUT_STATIONS, "times": heldout.times, "frequencies": heldout.frequencies,
                    "observed_power": heldout.power, "clean_power": clean.power, "predicted_power": prediction,
                    "used_during_fitting": False, "noise_seed": observed_config.seed+100003}, output/"heldout_prediction.pt")
        plot_stations(heldout, prediction, output/"heldout_stations.png",
                       figure_title="Predictions at four stations withheld entirely from fitting")
    return {"heldout_station_positions_m": HOLDOUT_STATIONS.tolist(),
            "heldout_station_count": len(HOLDOUT_STATIONS),
            "heldout_power_nmse": float(noisy_error), "heldout_clean_power_nmse": float(clean_error),
            "heldout_per_station_nmse": ((prediction-heldout.power).square().mean((1, 2))
                                          / heldout.power.square().mean((1, 2)).clamp_min(1e-20)).tolist(),
            "heldout_stations_used_during_fit": False}


def evaluate_saved_predictions(output: Path, conditions=CONDITIONS) -> None:
    """Reproduce prediction diagnostics from selected checkpoints without refitting."""
    for condition in conditions:
        root = output/condition
        manifest = json.loads((root/"results.json").read_text())
        for report in manifest["cases"]:
            case = root/f"calls_{report['events']}_stations_{report['stations']}"
            if (case/"heldout_prediction.pt").exists() and (case/"heldout_stations.png").exists():
                continue
            bundle = torch.load(case/"fit.pt", weights_only=True)
            if report["heldout_station_positions_m"] != HOLDOUT_STATIONS.tolist():
                raise ValueError("Prediction station layout differs from the saved evaluation")
            config, model = Config(**report["config"]), Config(**report["model_config"])
            scene = random_scene(config, report["events"], report["stations"])
            if not torch.allclose(scene.power, bundle["observed_power"], rtol=1e-5, atol=1e-10):
                raise ValueError("Saved observations do not reproduce for prediction audit")
            state = SimpleNamespace(**{k: bundle[k] for k in ("positions", "onsets", "amplitudes")})
            result = SimpleNamespace(state=state, splauts=Splauts(**bundle["splauts"]))
            measured = prediction_audit(scene, result, config, model, case)
            if abs(measured["heldout_power_nmse"] - report["heldout_power_nmse"]) > 1e-6:
                raise ValueError("Saved prediction metrics do not reproduce")


def plot_physics(config: Config, alpha: float, output: Path) -> None:
    call = CallSpec("moo", config.call_duration, 450., 120., harmonic_count=config.harmonic_count,
                    harmonic_rolloff=config.harmonic_rolloff)
    waveform = synthesize_call(call, config.sample_rate)
    power, times, frequencies = power_spectrogram(waveform, config)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow((10*(power/power.max()).clamp_min(1e-5).log10()).T.numpy(), origin="lower", aspect="auto",
                   extent=[float(times[0]), float(times[-1]), 0, config.sample_rate/2], vmin=-50, vmax=0)
    contour = instantaneous_frequency(call, times)
    for h in range(1, config.harmonic_count+1):
        axes[0].plot(times, h*contour, color="white", lw=0.7, ls="--", alpha=0.6)
    axes[0].set(title="Moo: smooth rise, sustained pitch, smooth fall", xlabel="Time (s)", ylabel="Frequency (Hz)")
    for distance in (50., 150., 300.):
        transfer = absorption_transmission(torch.tensor(distance), frequencies,
                                            replace(config, absorption_db_per_km=alpha), power=True)
        axes[1].plot(frequencies, 10*transfer.clamp_min(1e-30).log10(), label=f"{distance:g} m")
    axes[1].set(title=f"Spectral absorption: {alpha:g} dB/km at 1 kHz", xlabel="Frequency (Hz)", ylabel="Power loss (dB)")
    axes[1].legend()
    fig.tight_layout(); _save(fig, output / "moo_and_absorption.png")
    pcm = (waveform.numpy() / max(float(waveform.abs().max()), 1e-8) * 0.95 * 32767).astype("<i2")
    with wave.open(str(output / "moo_example.wav"), "wb") as file:
        file.setnchannels(1); file.setsampwidth(2); file.setframerate(config.sample_rate); file.writeframes(pcm.tobytes())


def summarize_ablation(output: Path, conditions=CONDITIONS) -> dict:
    datasets = {name: json.loads((output/name/"results.json").read_text()) for name in conditions}
    if any(d["completed"] != d["requested"] for d in datasets.values()):
        raise ValueError("Complete all requested conditions before summarizing")
    indexed = {name: {(r["events"], r["stations"]): r for r in data["cases"]} for name, data in datasets.items()}
    keys = sorted(indexed[conditions[0]])
    if any(set(data) != set(keys) for data in indexed.values()):
        raise ValueError("Ablation conditions must use the same case matrix")
    metrics = ("mean_position_error_m", "power_nmse", "mean_source_power_nmse", "heldout_power_nmse", "heldout_clean_power_nmse")
    summary = {}
    for name, rows in indexed.items():
        summary[name] = {"localized_within_10m": sum(sum(e <= 10 for e in r["position_errors_m"]) for r in rows.values()),
                         "total_sources": sum(r["events"] for r in rows.values()),
                         "all_decoders_unchanged": all(r["decoder_unchanged"] for r in rows.values()),
                         "all_test_calls_disjoint": all(r["test_calls_disjoint_from_training"] for r in rows.values()),
                         **{key: float(np.mean([r[key] for r in rows.values()])) for key in metrics}}
        errors = [error for r in rows.values() for error in r["position_errors_m"]]
        summary[name]["mean_instance_position_error_m"] = float(np.mean(errors))
        summary[name]["median_instance_position_error_m"] = float(np.median(errors))
    cases = []
    for key in keys:
        rows = [indexed[name][key] for name in conditions]
        if len({r["emission_fingerprint"] for r in rows}) != 1:
            raise ValueError("Paired emissions or geometry differ")
        if "modeled" in indexed and "ignored" in indexed:
            if indexed["modeled"][key]["observation_fingerprint"] != indexed["ignored"][key]["observation_fingerprint"]:
                raise ValueError("Modeled and ignored conditions must fit identical station mixtures")
        cases.append({"events": key[0], "stations": key[1], **{name: {k: indexed[name][key][k]
                       for k in (*metrics, "position_errors_m", "source_power_nmse")} for name in conditions}})
    first = next(iter(indexed[conditions[0]].values()))
    prior = json.loads((output/"prior"/"pretraining.json").read_text())
    family_names = Config(**prior["config"]).call_families
    family_errors = np.asarray(prior["validation_nmse"]).reshape(len(family_names), -1).mean(1)
    comparisons = {}
    for before, after in (("off", "modeled"), ("modeled", "ignored")):
        if before in indexed and after in indexed:
            comparisons[f"{before}_to_{after}"] = {}
            for metric in metrics:
                delta = np.asarray([indexed[after][key][metric]-indexed[before][key][metric] for key in keys])
                comparisons[f"{before}_to_{after}"][metric] = {
                    "mean_change": float(delta.mean()), "cases_with_lower_error": int((delta < 0).sum()),
                    "cases_with_higher_error": int((delta > 0).sum()), "total_cases": len(keys)}
    report = {"protocol": {"config": first["config"], "conditions": list(conditions),
                           "absorption_db_per_km": json.loads((output/"protocol.json").read_text())["absorption_db_per_km"],
                           "steps": first["steps"], "proposal_steps": first["proposal_steps"], "proposal_grid_size": first["proposal_grid_size"],
                           "same_frozen_decoder_for_all_conditions": True, "decoder_fingerprint": prior["decoder_fingerprint"],
                           "pretraining_examples": prior["training_examples"], "validation_examples": prior["validation_examples"],
                           "pretrain_steps": prior["pretrain_steps"], "isolated_validation_nmse": prior["mean_validation_nmse"],
                           "isolated_validation_nmse_by_family": dict(zip(family_names, family_errors.tolist())),
                           "training_seed": prior["training_seed"], "validation_seed": prior["validation_seed"],
                           "prediction": "four interior stations withheld entirely from fitting; independent noise seed",
                           "heldout_station_positions_m": HOLDOUT_STATIONS.tolist(),
                           "heldout_noise_seed": first["scene_seed"]+100003,
                           "event_count_and_duration_supplied": True,
                           "absorption_law_supplied_to_modeled_condition": True,
                           "controls": "identical emissions, positions, onsets, amplitudes, noise draws, prior and optimization budgets",
                           "averaging": "NMSE and mean_position_error_m are case means; mean/median_instance_position_error_m pool source instances",
                           "proposal_absorption": "distance-binned spectral templates; exact continuous transfer during gradient refinement",
                           "limitations": "one scene seed; synthetic power-law absorption; supplied event count and duration; waveform/STFT versus additive-power approximation"},
              "summary": summary, "paired_comparisons": comparisons, "cases": cases}
    _write_json(output/"ablation.json", report)
    fig, axes = plt.subplots(1, 4, figsize=(19, 7))
    labels = [f"{k} calls / {m} stations" for k, m in keys]
    for ax, metric, title in zip(axes, ("fraction_within_10m", "mean_source_power_nmse", "power_nmse", "heldout_clean_power_nmse"),
                                  ("Localized within 10 m", "Emitted-source NMSE", "Fitted-station NMSE", "Unseen-station NMSE (clean signal)")):
        y = np.arange(len(keys))
        width = 0.8 / len(conditions)
        for j, name in enumerate(conditions):
            ax.barh(y + (j-(len(conditions)-1)/2)*width, [indexed[name][key][metric] for key in keys], height=width, label=name)
        ax.set(yticks=y, yticklabels=labels, xlabel=title)
        ax.legend(fontsize=8)
    fig.suptitle("Paired moo/harmonic calls: absorption off, modeled correctly, or ignored")
    fig.tight_layout(); _save(fig, output/"ablation.png")
    return report


def run_ablation(config: Config, output: Path, alpha: float = 10., event_counts=(2, 3, 4, 5),
                 station_counts=(4, 6, 8), conditions=CONDITIONS, steps: int = 400,
                 proposal_steps: int = 80, proposal_grid_size: int = 17, pretrain_steps: int = 3000,
                 checkpoint: Path | None = None, resume: bool = False) -> dict:
    if not alpha > 0 or not conditions or any(name not in CONDITIONS for name in conditions):
        raise ValueError("Use positive absorption and supported conditions")
    if config.seed in (23, 29):
        raise ValueError("Mixture seed must be separate from training/validation seeds")
    output.mkdir(parents=True, exist_ok=True)
    if resume and checkpoint is None and (output/"prior"/"prior.pt").exists():
        checkpoint = output/"prior"/"prior.pt"
    prior_config = replace(config, absorption_db_per_km=0.)
    prior, metadata = prepare_prior(prior_config, output/"prior", 32, pretrain_steps, 23, 29,
                                    25*config.frequency_scale, checkpoint)
    if config.seed in (metadata["training_seed"], metadata["validation_seed"]):
        raise ValueError("Mixture seed overlaps checkpoint training/validation")
    protocol = {"config": asdict(config), "absorption_db_per_km": alpha, "conditions": list(conditions),
                "event_counts": list(event_counts), "station_counts": list(station_counts), "steps": steps,
                "proposal_steps": proposal_steps, "proposal_grid_size": proposal_grid_size,
                "decoder_fingerprint": prior.fingerprint}
    if resume and (output/"protocol.json").exists():
        saved = json.loads((output/"protocol.json").read_text())
        normalized = json.loads(json.dumps(protocol))
        if saved != normalized:
            raise ValueError("Resume settings differ from the saved ablation protocol")
    _write_json(output/"protocol.json", protocol)
    plot_physics(config, alpha, output)
    cases = list(itertools.product(event_counts, station_counts))
    for condition in conditions:
        true_config = replace(config, absorption_db_per_km=0. if condition == "off" else alpha)
        model_config = replace(true_config, absorption_db_per_km=0. if condition == "ignored" else true_config.absorption_db_per_km)
        root = output/condition; root.mkdir(exist_ok=True)
        rows = []
        for index, (count, stations) in enumerate(cases):
            print(f"Absorption {condition} {index+1}/{len(cases)}: {count} calls, {stations} stations", flush=True)
            case = root/f"calls_{count}_stations_{stations}"; case.mkdir(exist_ok=True)
            required = ("metrics.json", "fit.pt", "components.png", "station_mixtures.png", "evaluation.png", "losses.png")
            if resume and all((case/name).exists() for name in required):
                report = json.loads((case/"metrics.json").read_text())
                if (Config(**report["config"]) != true_config or Config(**report["model_config"]) != model_config
                        or report["decoder_fingerprint_before"] != prior.fingerprint):
                    raise ValueError(f"Incompatible retained case: {case}")
                rows.append(report); _save_sweep(root, rows, len(cases))
                print("  retained completed fit", flush=True)
                continue
            scene = random_scene(true_config, count, stations)
            observed = MixtureObservations(scene.power, scene.stations, scene.times, scene.frequencies)
            started = time.perf_counter()
            result = fit_frozen_prior(observed, prior, model_config, count, steps, proposal_steps, proposal_grid_size)
            elapsed = time.perf_counter()-started
            # All test labels become evaluation inputs only after model selection.
            trained = [asdict(CallSpec(**c)) for c in metadata["training_call_specs"]]
            disjoint = all(asdict(e.call) not in trained for e in scene.events)
            unchanged = decoder_fingerprint(prior.generator) == prior.fingerprint
            if not disjoint or not unchanged:
                raise RuntimeError("Training split or frozen-decoder audit failed")
            evaluation = prediction_audit(scene, result, true_config, model_config, case)
            details = {"variant": "frozen_pretrained", "condition": condition, "model_config": asdict(model_config),
                       "steps": steps, "proposal_steps": proposal_steps, "proposal_grid_size": proposal_grid_size,
                       "decoder_fingerprint_before": prior.fingerprint, "decoder_fingerprint_after": result.decoder_fingerprint,
                       "decoder_unchanged": unchanged, "decoder_trainable_parameters": 0,
                       "test_calls_disjoint_from_training": disjoint, "inferred_families": [config.call_families[int(k)] for k in result.selected_families],
                       "emission_fingerprint": _emission_fingerprint(scene),
                       "observation_fingerprint": hashlib.sha256(scene.power.contiguous().numpy().tobytes()).hexdigest(),
                       **evaluation}
            rows.append(record_mixture_fit(scene, result, true_config, case, details, elapsed))
            _save_sweep(root, rows, len(cases))
        evaluate_saved_sources(root)
    evaluate_saved_predictions(output, conditions)
    return summarize_ablation(output, conditions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/moo_attenuation"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Regenerate prediction artifacts and summary without fitting")
    parser.add_argument("--events", type=int, nargs="+", choices=(2, 3, 4, 5), default=[2, 3, 4, 5])
    parser.add_argument("--stations", type=int, nargs="+", choices=(4, 6, 8), default=[4, 6, 8])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--absorption-db-per-km", type=float, default=10.)
    parser.add_argument("--absorption-power", type=float, default=2.)
    parser.add_argument("--absorption-distance-bins", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--proposal-steps", type=int, default=80)
    parser.add_argument("--proposal-grid-size", type=int, default=17)
    parser.add_argument("--pretrain-steps", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or min(args.steps, args.proposal_steps, args.pretrain_steps) < 0 or args.proposal_grid_size < 2:
        parser.error("Use positive threads, nonnegative iterations and grid >=2")
    torch.set_flush_denormal(True); torch.set_num_threads(args.threads)
    if args.audit_only:
        evaluate_saved_predictions(args.output, args.conditions)
        report = summarize_ablation(args.output, args.conditions)
        print(json.dumps(report["summary"], indent=2))
        return
    config = Config(call_duration=0.6, scene_duration=3., harmonic_count=4, frequency_scale=0.3,
                    splauts=96, call_families=("moo", "tonal", "sweep", "pulse"), noise_std=args.noise,
                    seed=args.seed, absorption_power=args.absorption_power)
    config = replace(config, absorption_distance_bins=args.absorption_distance_bins)
    report = run_ablation(config, args.output, args.absorption_db_per_km, args.events, args.stations,
                           args.conditions, args.steps, args.proposal_steps, args.proposal_grid_size,
                           args.pretrain_steps, args.prior, args.resume)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
