"""Reproducible demos using waveform observations and known event associations."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from .config import Config, station_layout
from .model.events import EventState
from .model.generator import evaluate_field
from .simulator.calls import CallSpec
from .simulator.scene import SourceEvent, random_scene, simulate_scene
from .training.localize import initialize_positions, localize, spatial_loss_grid
from .training.train_calls import train_calls
from .visualization.plots import (plot_calls, plot_comparison, plot_landscape,
                                  plot_losses, plot_map, plot_stations)


def _truth(scene, training) -> EventState:
    return EventState(torch.stack([e.position for e in scene.events]),
                      torch.tensor([e.onset for e in scene.events]),
                      torch.tensor([e.amplitude ** 2 for e in scene.events]),
                      training.durations, training.latents)


def _metrics(result) -> dict:
    return {"initial_error_m": result.location_errors[0].tolist(),
            "final_error_m": result.location_errors[-1].tolist(),
            "initial_reconstruction_mse": result.reconstruction_losses[0],
            "final_reconstruction_mse": result.reconstruction_losses[-1],
            "estimated_positions_m": result.state.positions.tolist(),
            "estimated_onsets_s": result.state.onsets.tolist(),
            "estimated_power_scales": result.state.amplitudes.tolist()}


def run_demo(config: Config, output: Path, events: int = 1, stations: int = 4,
             layout: str = "corners", call_type: str | None = None,
             train_steps: int = 600, localization_steps: int = 250,
             grid_size: int = 25, refine: bool = False, coarse_grid_size: int = 13) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    scene = random_scene(config, events, stations, layout, call_type)
    print(f"Training {events} associated call(s) on CPU...", flush=True)
    training = train_calls([e.call for e in scene.events], config, train_steps)
    truth = _truth(scene, training)
    initial = truth.detach()
    # A neutral centre initialization, independent of labels and event identity.
    initial.positions = torch.full((events, 2), config.area_size / 2)
    neutral_positions = initial.positions.clone()
    if events > 1 and coarse_grid_size:
        print(f"Initializing associated events with {coarse_grid_size}×{coarse_grid_size} reconstruction grids...", flush=True)
        initial = initialize_positions(training.generator, initial, scene.power, scene.stations,
                                       scene.times, scene.frequencies, config, coarse_grid_size)
    common = dict(generator=training.generator, observation=scene.power, stations=scene.stations,
                  times=scene.times, frequencies=scene.frequencies, config=config,
                  steps=localization_steps, true_positions=truth.positions)
    print("Localizing with and without position supervision...", flush=True)
    unsupervised = localize(initial=initial, **common)
    supervised = localize(initial=initial, position_weight=1.0, **common)
    runs = {"No position labels": unsupervised, "Position labels (weight=1)": supervised}
    if refine:
        state = unsupervised.state
        for stage, label in enumerate(("+ onset", "+ amplitude", "+ embedding")):
            result = localize(initial=state, optimize_onsets=True, optimize_amplitudes=stage >= 1,
                              optimize_latents=stage >= 2, **common)
            runs[label] = result
            state = result.state
    plot_map(scene, unsupervised, config.area_size, output / "map.png")
    plot_calls(training, output / "calls.png")
    plot_stations(scene, unsupervised.prediction, output / "stations.png")
    plot_losses(training, runs, output / "losses.png")
    landscapes = []
    print(f"Evaluating {events} conditional {grid_size}×{grid_size} spatial loss grid(s)...", flush=True)
    for k in range(events):
        axis, losses = spatial_loss_grid(training.generator, unsupervised.state, scene.power,
                                         scene.stations, scene.times, scene.frequencies, config, grid_size, k)
        plot_landscape(axis, losses, scene, unsupervised, output / f"landscape_{k+1}.png", k)
        landscapes.append(losses)
    report = {"config": asdict(config), "stations_m": scene.stations.tolist(),
              "calls": [asdict(e.call) for e in scene.events],
              "true_positions_m": truth.positions.tolist(), "true_onsets_s": truth.onsets.tolist(),
              "true_power_scales": truth.amplitudes.tolist(),
              "call_mse_initial": training.losses[0], "call_mse_final": training.losses[-1],
              "initialization": {"neutral_positions_m": neutral_positions.tolist(),
                                 "gradient_start_positions_m": initial.positions.tolist(),
                                 "coarse_grid_size": coarse_grid_size if events > 1 else 0,
                                 "uses_position_labels": False},
              "runs": {name: _metrics(result) for name, result in runs.items()},
              "observation_model": "Fractionally delayed waveforms; additive power renderer is approximate",
              "known_at_initialization": ["event associations", "call embeddings", "onsets", "source amplitudes"]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Tensor-only bundle can be read using torch.load(..., weights_only=True).
    torch.save({"generator": training.generator.state_dict(), "latents": training.latents,
                "durations": training.durations, "stations": scene.stations,
                "waveforms": scene.waveforms, "observed_power": scene.power,
                "source_power": training.targets, "times": scene.times, "frequencies": scene.frequencies,
                "predicted_power": unsupervised.prediction, "trajectory": unsupervised.trajectories,
                "training_losses": torch.tensor(training.losses),
                "runs": {name: {"trajectory": result.trajectories,
                                 "reconstruction_losses": torch.tensor(result.reconstruction_losses),
                                 "objective_losses": torch.tensor(result.objective_losses),
                                 "location_errors": result.location_errors,
                                 "positions": result.state.positions, "onsets": result.state.onsets,
                                 "amplitudes": result.state.amplitudes, "latents": result.state.latents}
                         for name, result in runs.items()},
                "spatial_axis": axis, "spatial_losses": torch.stack(landscapes),
                "generated_source_power": evaluate_field(training.generator(training.latents, training.durations),
                                                          training.times, training.frequencies).detach()},
               output / "experiment.pt")
    print(f"Position errors without labels (m): {unsupervised.location_errors[-1].tolist()}", flush=True)
    return report


def run_comparison(config: Config, output: Path, train_steps: int = 600,
                   localization_steps: int = 250, grid_size: int = 15) -> list[dict]:
    """Change one variable at a time from a fixed four-station/no-noise tonal baseline."""
    output.mkdir(parents=True, exist_ok=True)
    calls = [CallSpec(kind, config.call_duration, 1450., 260.) for kind in config.call_families]
    training = train_calls(calls, config, train_steps)
    cases = [("baseline", 4, "corners", 0., 0),
             ("irregular perimeter", 4, "irregular", 0., 0),
             ("clustered stress", 4, "clustered", 0., 0),
             ("low noise", 4, "corners", 0.0001, 0),
             ("moderate noise", 4, "corners", 0.0005, 0),
             ("sweep", 4, "corners", 0., 1), ("pulse train", 4, "corners", 0., 2),
             ("six stations", 6, "corners", 0., 0), ("eight stations", 8, "corners", 0., 0)]
    rows = []
    position = torch.tensor([[120., 165.]]) * config.area_size / 300
    initial_position = torch.tensor([[70., 220.]]) * config.area_size / 300
    for index, (name, count, layout, noise, call_index) in enumerate(cases):
        print(f"Comparison {index+1}/{len(cases)}: {name}", flush=True)
        case_config = replace(config, noise_std=noise)
        event = SourceEvent(position[0], 0.2, calls[call_index])
        scene = simulate_scene([event], station_layout(count, layout, config.area_size), case_config)
        state = EventState(initial_position.clone(), torch.tensor([0.2]), torch.ones(1),
                           training.durations[call_index:call_index+1], training.latents[call_index:call_index+1])
        result = localize(training.generator, state, scene.power, scene.stations,
                          scene.times, scene.frequencies, case_config, steps=localization_steps,
                          true_positions=position)
        axis, losses = spatial_loss_grid(training.generator, result.state, scene.power,
                                         scene.stations, scene.times, scene.frequencies, case_config, grid_size)
        plot_landscape(axis, losses, scene, result, output / f"landscape_{index+1}.png")
        rows.append({"case": name, "stations": count, "layout": layout, "noise_std": noise,
                     "call_type": calls[call_index].kind, "error_m": float(result.location_errors[-1, 0]),
                     "reconstruction_mse": result.reconstruction_losses[-1]})
    plot_comparison(rows, output / "comparison.png")
    (output / "comparison.json").write_text(json.dumps({"config": asdict(config), "cases": rows}, indent=2), encoding="utf-8")
    return rows
