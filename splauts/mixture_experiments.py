"""Mixture-only experiments and post-fit evaluation of the requested 4×3 matrix."""

from dataclasses import asdict
import csv
import itertools
import json
from pathlib import Path
import time

import torch

from .config import Config
from .model.generator import Splauts, evaluate_field
from .simulator.scene import random_scene
from .training.mixtures import MixtureObservations, fit_mixtures, interpolate_power, mixture_losses
from .visualization.mixtures import plot_components, plot_evaluation, plot_fit_losses, plot_source_evaluation, plot_sweep
from .visualization.plots import plot_stations


def match_positions(estimated: torch.Tensor, truth: torch.Tensor) -> list[int]:
    """Evaluation ONLY: exhaustive assignment, feasible for at most five events."""
    if estimated.shape != truth.shape or estimated.ndim != 2 or estimated.shape[1] != 2:
        raise ValueError("Estimated and true positions must both have shape [events, 2]")
    if not 1 <= len(truth) <= 5:
        raise ValueError("Position matching supports 1–5 events")
    distances = torch.cdist(truth, estimated)
    return list(min(itertools.permutations(range(len(truth))),
                    key=lambda permutation: sum(float(distances[k, v]) for k, v in enumerate(permutation))))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def _save_sweep(output: Path, rows: list[dict], requested: int) -> None:
    _write_json(output / "results.json", {"completed": len(rows), "requested": requested, "cases": rows})
    keys = ("events", "stations", "mean_position_error_m", "max_position_error_m", "fraction_within_10m",
            "power_nmse", "compressed_loss", "mean_source_power_nmse", "mean_emission_centroid_error_s", "fit_seconds")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_sweep(rows, output / "summary.png")


@torch.no_grad()
def evaluate_saved_sources(output: Path) -> list[dict]:
    """Post-fit audit of emitted fields; never retrains or selects fitted models.

    Also usable to add source-quality evaluation to an existing saved sweep.
    Clean sources are regenerated solely for evaluation, after verifying that
    the regenerated station mixtures match the fitted observations.
    """
    manifest = json.loads((output / "results.json").read_text(encoding="utf-8"))
    rows = manifest["cases"]
    for report in rows:
        case = output / f"calls_{report['events']}_stations_{report['stations']}"
        bundle = torch.load(case / "fit.pt", weights_only=True)
        scene = random_scene(Config(**report["config"]), report["events"], report["stations"])
        if not torch.allclose(bundle["observed_power"], scene.power, rtol=1e-5, atol=1e-10):
            raise ValueError("Regenerated mixtures do not match saved observations; evaluation truth is incompatible")
        splauts = Splauts(**bundle["splauts"])
        truth_fields, estimated_fields = [], []
        for truth_index, component in enumerate(report["assignment_truth_to_component"]):
            event = scene.events[truth_index]
            source_dt = float(scene.source_times[truth_index][1] - scene.source_times[truth_index][0])
            query = (scene.times - event.onset)[None, None]
            target = interpolate_power(scene.source_power[truth_index][None], query, source_dt)[0, 0] * event.amplitude ** 2
            single = Splauts(**{name: tensor[component:component+1] for name, tensor in vars(splauts).items()})
            predicted = (evaluate_field(single, scene.times - bundle["onsets"][component], scene.frequencies)[0]
                         * bundle["amplitudes"][component])
            truth_fields.append(target)
            estimated_fields.append(predicted)
        truth, estimated = torch.stack(truth_fields), torch.stack(estimated_fields)
        errors = ((estimated - truth).square().mean((1, 2)) / truth.square().mean((1, 2)).clamp_min(1e-20))
        def centroids(fields):
            return (fields * scene.times[None, :, None]).sum((1, 2)) / fields.sum((1, 2)).clamp_min(1e-20)
        timing = (centroids(truth) - centroids(estimated)).abs()
        report.update(source_power_nmse=errors.tolist(), mean_source_power_nmse=float(errors.mean()),
                      emission_centroid_errors_s=timing.tolist(), mean_emission_centroid_error_s=float(timing.mean()))
        _write_json(case / "metrics.json", report)
        plot_source_evaluation(truth, estimated, scene.times, scene.frequencies, case / "source_evaluation.png")
    _save_sweep(output, rows, manifest["requested"])
    return rows


def run_mixture_case(config: Config, output: Path, event_count: int, station_count: int,
                     warmup_steps: int = 250, steps: int = 400, proposal_grid_size: int = 13,
                     fit_seed: int = 101, geometry_steps: int = 75) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    scene = random_scene(config, event_count, station_count)
    # This is the complete data boundary. No source waveform, clean source STFT,
    # call specification, position, onset, amplitude or event association crosses it.
    observed = MixtureObservations(scene.power, scene.stations, scene.times, scene.frequencies)
    print(f"Mixture-only fit: {event_count} calls, {station_count} stations", flush=True)
    started = time.perf_counter()
    result = fit_mixtures(observed, config, event_count, warmup_steps=warmup_steps, steps=steps,
                          geometry_steps=geometry_steps, proposal_grid_size=proposal_grid_size, fit_seed=fit_seed)
    elapsed = time.perf_counter() - started
    protocol = {"variant": "mixture_only", "fit_seed": fit_seed, "warmup_steps": warmup_steps, "steps": steps,
                "geometry_steps": geometry_steps, "proposal_grid_size": proposal_grid_size,
                "decoder_initialization": "fresh random MLP, warmed on mixture-derived back-projections only"}
    return record_mixture_fit(scene, result, config, output, protocol, elapsed)


def record_mixture_fit(scene, result, config: Config, output: Path, protocol: dict, elapsed: float) -> dict:
    """Shared post-inversion metrics and artifacts for both decoder variants."""
    event_count, station_count = len(scene.events), len(scene.stations)
    observed = MixtureObservations(scene.power, scene.stations, scene.times, scene.frequencies)
    # Source labels first become accessible to this path AFTER fitting and model selection.
    truth = torch.stack([event.position for event in scene.events])
    assignment = match_positions(result.state.positions, truth)
    ordered = result.state.positions[assignment]
    error = (ordered - truth).norm(dim=-1)
    history_errors = (result.trajectories[:, assignment] - truth[None]).norm(dim=-1).numpy()
    power, compressed = mixture_losses(result.prediction, observed.power)
    per_station = ((result.prediction - observed.power).square().mean((1, 2))
                   / observed.power.square().mean((1, 2))).tolist()
    report = {"events": event_count, "stations": station_count, "scene_seed": config.seed,
              "config": asdict(config),
              "fitting_inputs": ["station mixture power", "station coordinates", "time/frequency axes", "call count"],
              "priors": {"duration_s": config.call_duration, "proposal_bandwidth_hz": 500.,
                         "area_size_m": config.area_size, "source_power_scale": "learned RMS power; decoder fields have unit RMS"},
              "withheld_during_fit": ["source waveforms", "source spectrograms", "call identities", "event associations",
                                      "source positions", "emission onsets", "source amplitudes"],
              "model_selection": "minimum mixture objective; no ground-truth selection",
              "assignment_truth_to_component": assignment,
              "true_positions_m": truth.tolist(), "estimated_positions_m": result.state.positions.tolist(),
              "true_onsets_s": [event.onset for event in scene.events], "estimated_onsets_s": result.state.onsets.tolist(),
              "estimated_source_rms_power": result.state.amplitudes.tolist(),
              "call_types_evaluation_only": [event.call.kind for event in scene.events],
              "position_errors_m": error.tolist(), "mean_position_error_m": float(error.mean()),
              "max_position_error_m": float(error.max()), "fraction_within_10m": float((error <= 10).float().mean()),
              "power_nmse": float(power), "compressed_loss": float(compressed), "per_station_power_nmse": per_station,
              "initial_power_nmse": result.power_losses[0], "best_step": result.best_step, "fit_seconds": elapsed}
    report.update(protocol)
    torch.save({"generator": result.generator.state_dict(), "latents": result.state.latents,
                "positions": result.state.positions, "onsets": result.state.onsets, "durations": result.state.durations,
                "amplitudes": result.state.amplitudes, "observed_power": observed.power,
                "prediction": result.prediction, "stations": observed.stations, "times": observed.times,
                "frequencies": observed.frequencies, "proposed_positions": result.proposals.positions,
                "proposed_onsets": result.proposals.onsets, "provisional_fields": result.proposals.fields,
                "relative_times": result.proposals.relative_times, "trajectories": result.trajectories,
                "onset_history": result.onset_history, "objectives": torch.tensor(result.objectives),
                "power_losses": torch.tensor(result.power_losses), "compressed_losses": torch.tensor(result.compressed_losses),
                "splauts": vars(result.splauts)}, output / "fit.pt")
    _write_json(output / "metrics.json", report)
    plot_components(result, scene.frequencies, output / "components.png")
    plot_stations(scene, result.prediction, output / "station_mixtures.png")
    plot_fit_losses(result, output / "losses.png")
    plot_evaluation(scene, result, assignment, history_errors, output / "evaluation.png")
    print(f"  Evaluation only: errors(m)={[round(v, 2) for v in error.tolist()]}; NMSE={float(power):.4f}; fit={elapsed:.1f}s", flush=True)
    return report


def run_mixture_sweep(config: Config, output: Path, event_counts=(2, 3, 4, 5), station_counts=(4, 6, 8),
                      warmup_steps: int = 250, steps: int = 400, proposal_grid_size: int = 13,
                      fit_seed: int = 101, geometry_steps: int = 75) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = list(itertools.product(event_counts, station_counts))
    for index, (events, stations) in enumerate(cases):
        print(f"Case {index+1}/{len(cases)}", flush=True)
        report = run_mixture_case(config, output / f"calls_{events}_stations_{stations}", events, stations,
                                  warmup_steps, steps, proposal_grid_size, fit_seed, geometry_steps)
        rows.append(report)
        # Persist completed cases so a long sweep is inspectable while it runs.
        _save_sweep(output, rows, len(cases))
    return evaluate_saved_sources(output)
