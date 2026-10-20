"""Pretrain once on isolated calls, freeze, and invert independent scene mixtures."""

from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time

import torch

from .config import Config
from .mixture_experiments import _save_sweep, _write_json, evaluate_saved_sources, record_mixture_fit
from .model.call_prior import acoustic_ranges, acoustic_values
from .simulator.calls import CallSpec
from .simulator.scene import random_scene
from .training.frozen import fit_frozen_prior
from .training.mixtures import MixtureObservations
from .training.pretrain import (decoder_fingerprint, evaluate_prior_calls, harmonic_reconstruction_metrics, load_frozen_prior,
                                pretrain_call_prior)
from .visualization.pretrained import plot_prior_comparison, plot_prior_quality


def prepare_prior(config: Config, output: Path, per_family: int, pretrain_steps: int,
                   pretrain_seed: int, validation_seed: int, bank_step: float,
                   checkpoint: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        prior = load_frozen_prior(checkpoint, config)
        bundle = torch.load(checkpoint, weights_only=True)
        torch.save(bundle, output / "prior.pt")
        _write_json(output / "pretraining.json", bundle["metadata"])
        return prior, bundle["metadata"]
    print(f"Pretraining on {per_family * len(config.call_families)} isolated calls (seed {pretrain_seed})...", flush=True)
    started = time.perf_counter()
    training = pretrain_call_prior(config, per_family, pretrain_steps, pretrain_seed,
                                   validation_seed, frequency_step=bank_step)
    prior = training.prior
    validation_target, validation_prediction = evaluate_prior_calls(prior, training.validation_calls, config)
    metadata = {"config": asdict(config), "training_seed": pretrain_seed, "validation_seed": validation_seed,
                "training_examples": len(training.calls), "validation_examples": len(training.validation_calls),
                "pretrain_steps": pretrain_steps, "bank_frequency_step_hz": bank_step,
                "bank_templates": len(prior.bank_codes), "acoustic_ranges": acoustic_ranges(config),
                "harmonic_structure": {"count": config.harmonic_count, "amplitude_rolloff": config.harmonic_rolloff,
                                       "energy_normalization": "sum of squared harmonic amplitudes equals one",
                                       "atomic_event": "one code, position and onset for all harmonic bands"},
                "training_call_specs": [asdict(call) for call in training.calls],
                "validation_call_specs": [asdict(call) for call in training.validation_calls],
                "decoder_fingerprint": prior.fingerprint, "training_nmse": training.train_nmse,
                "validation_nmse": training.validation_nmse, "training_history": training.training_history,
                "mean_training_nmse": sum(training.train_nmse) / len(training.train_nmse),
                "mean_validation_nmse": sum(training.validation_nmse) / len(training.validation_nmse),
                "validation_harmonic_metrics": harmonic_reconstruction_metrics(prior, training.validation_calls, config),
                "validation_uses_known_codes": True, "validation_selects_checkpoint": False,
                "pretraining_seconds": time.perf_counter()-started}
    torch.save({"config": asdict(config), "generator": prior.generator.state_dict(), "bank_codes": prior.bank_codes,
                "relative_times": prior.relative_times, "frequencies": prior.frequencies,
                "fingerprint": prior.fingerprint, "metadata": metadata}, output / "prior.pt")
    _write_json(output / "pretraining.json", metadata)
    plot_prior_quality(training, validation_target, validation_prediction, output)
    print(f"  train NMSE={metadata['mean_training_nmse']:.4f}, held-out isolated NMSE={metadata['mean_validation_nmse']:.4f}", flush=True)
    return prior, metadata


def run_pretrained_sweep(config: Config, output: Path, event_counts=(2, 3, 4, 5), station_counts=(4, 6, 8),
                         pretrain_steps: int = 2000, per_family: int = 32, steps: int = 400,
                         proposal_steps: int = 80, proposal_grid_size: int = 25,
                         pretrain_seed: int = 23, validation_seed: int = 29,
                         bank_step: float = 25., checkpoint: Path | None = None,
                         pretrain_only: bool = False, resume: bool = False) -> list[dict]:
    if not pretrain_only and checkpoint is None and config.seed in (pretrain_seed, validation_seed):
        raise ValueError("Use independent pretraining, validation and mixture-scene seeds")
    prior, metadata = prepare_prior(config, output, per_family, pretrain_steps, pretrain_seed,
                                    validation_seed, bank_step, checkpoint)
    if not pretrain_only and config.seed in (metadata["training_seed"], metadata["validation_seed"]):
        raise ValueError("Mixture seed must differ from the checkpoint's training/validation seeds")
    if pretrain_only:
        return []
    rows, cases = [], list(itertools.product(event_counts, station_counts))
    for index, (events, stations) in enumerate(cases):
        print(f"Frozen-prior case {index+1}/{len(cases)}: {events} calls, {stations} stations", flush=True)
        case_output = output / f"calls_{events}_stations_{stations}"
        case_output.mkdir(parents=True, exist_ok=True)
        required = ("metrics.json", "fit.pt", "components.png", "station_mixtures.png", "losses.png", "evaluation.png")
        if resume and all((case_output / name).exists() for name in required):
            report = json.loads((case_output / "metrics.json").read_text(encoding="utf-8"))
            expected = {"steps": steps, "proposal_steps": proposal_steps,
                        "proposal_grid_size": proposal_grid_size, "decoder_fingerprint_before": prior.fingerprint,
                        "decoder_unchanged": True, "events": events, "stations": stations}
            if (Config(**report["config"]) != config or any(report.get(key) != value for key, value in expected.items())
                    or not (case_output / "fit.pt").exists()):
                raise ValueError(f"Saved case is incompatible with resume: {case_output}")
            # evaluate_saved_sources verifies regenerated observations against
            # every retained fit before completing the resumed sweep.
            rows.append(report)
            _save_sweep(output, rows, len(cases))
            print("  Resuming: retained completed fit", flush=True)
            continue
        scene = random_scene(config, events, stations)
        observed = MixtureObservations(scene.power, scene.stations, scene.times, scene.frequencies)
        started = time.perf_counter()
        result = fit_frozen_prior(observed, prior, config, events, steps, proposal_steps, proposal_grid_size)
        elapsed = time.perf_counter() - started
        # Split audit and all test-call labels are consulted only after inversion.
        trained_specs = [asdict(CallSpec(**spec)) for spec in metadata["training_call_specs"]]
        disjoint = all(asdict(event.call) not in trained_specs for event in scene.events)
        if not disjoint:
            raise RuntimeError("A test call duplicates an isolated pretraining example")
        unchanged = decoder_fingerprint(prior.generator) == prior.fingerprint
        if not unchanged:
            raise RuntimeError("Decoder changed during the held-out mixture sweep")
        protocol = {"variant": "frozen_pretrained", "steps": steps, "proposal_steps": proposal_steps,
                    "proposal_grid_size": proposal_grid_size, "pretraining_seed": metadata["training_seed"],
                    "pretraining_checkpoint": "../prior.pt", "training_examples": metadata["training_examples"],
                    "decoder_initialization": "supervised isolated-call pretraining; frozen throughout inversion",
                    "decoder_fingerprint_before": prior.fingerprint, "decoder_fingerprint_after": result.decoder_fingerprint,
                    "decoder_unchanged": unchanged, "test_calls_disjoint_from_training": disjoint,
                    "fitting_inputs": ["station mixtures", "station coordinates", "time/frequency axes", "call count",
                                       "frozen pretrained decoder", "generic acoustic template bank"],
                    "inversion_trainable_parameters": ["position", "onset", "RMS power", "frequency", "bandwidth", "modulation rate", "pulse rate"],
                    "decoder_trainable_parameters": 0,
                    "inferred_families": [config.call_families[int(k)] for k in result.selected_families],
                    "estimated_acoustic_values": acoustic_values(result.state.latents, config).tolist(),
                    "bank_indices": result.bank_indices,
                    "initialization": "frozen neural template matching pursuit, followed by continuous event refinement",
                    "priors": {"duration_s": config.call_duration, "call_count_supplied": True,
                               "family": "selected from configured families using mixtures, then held fixed",
                               "acoustic_ranges": acoustic_ranges(config), "source_power_scale": "learned event RMS power"}}
        report = record_mixture_fit(scene, result, config, case_output, protocol, elapsed)
        rows.append(report)
        _save_sweep(output, rows, len(cases))
    rows = evaluate_saved_sources(output)
    baseline_path = Path(__file__).resolve().parent.parent / "Design/Experiments/mixture-only-seed7.json"
    if baseline_path.exists() and config.seed == 7:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        observation_keys = ("sample_rate", "n_fft", "hop_length", "scene_duration", "call_duration",
                            "area_size", "sound_speed", "distance_offset", "noise_std", "seed",
                            "harmonic_count", "harmonic_rolloff", "frequency_scale", "call_families",
                            "absorption_db_per_km", "absorption_power")
        baseline_config = Config(**baseline["protocol"]["config"])
        if all(getattr(config, k) == getattr(baseline_config, k) for k in observation_keys):
            plot_prior_comparison(rows, baseline["cases"], output / "comparison.png")
    return rows
