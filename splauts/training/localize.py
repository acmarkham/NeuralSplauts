"""Propagation-based direct optimization with optional, explicitly weighted labels."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import Config, validate_stations
from ..model.events import EventParameters, EventState
from ..model.generator import SplautGenerator
from ..model.renderer import render


@dataclass
class LocalizationResult:
    state: EventState
    prediction: Tensor
    trajectories: Tensor  # [iterations + 1, events, xy]
    reconstruction_losses: list[float]  # MSE / mean(observed power squared)
    objective_losses: list[float]
    location_errors: Tensor | None  # [iterations + 1, events], metres


def normalized_mse(prediction: Tensor, observation: Tensor) -> Tensor:
    return (prediction - observation).square().mean() / observation.square().mean().clamp_min(1e-20)


def localize(generator: SplautGenerator, initial: EventState, observation: Tensor,
             stations: Tensor, times: Tensor, frequencies: Tensor, config: Config,
             steps: int = 250, learning_rate: float = 0.04,
             optimize_onsets: bool = False, optimize_amplitudes: bool = False,
             optimize_latents: bool = False, true_positions: Tensor | None = None,
             true_onsets: Tensor | None = None, position_weight: float = 0.0,
             onset_weight: float = 0.0) -> LocalizationResult:
    """Labels only affect fitting when their corresponding weight is nonzero.

    The generator is frozen via detached functional parameters. Ground-truth
    positions may be supplied solely for reporting errors, even during inference.
    """
    validate_stations(stations)
    if steps < 0 or learning_rate <= 0 or min(position_weight, onset_weight) < 0:
        raise ValueError("Invalid optimization steps, learning rate or supervision weights")
    if (position_weight and true_positions is None) or (onset_weight and true_onsets is None):
        raise ValueError("Nonzero supervised weights require ground-truth labels")
    if observation.shape != (len(stations), len(times), len(frequencies)):
        raise ValueError("Observation must have shape [stations, time, frequency]")
    if not torch.isfinite(observation).all() or observation.square().mean() <= 0:
        raise ValueError("Observation must be finite and contain signal energy")
    parameters = EventParameters(initial.detach(), config.area_size, float(times[-1]),
                                 optimize_onsets, optimize_amplitudes, optimize_latents)
    groups = [{"params": [parameters.position_logits], "lr": learning_rate}]
    for parameter in (parameters.onset_logits, parameters.log_amplitudes, parameters.latents):
        if parameter.requires_grad:
            groups.append({"params": [parameter], "lr": learning_rate / 4})
    optimizer = torch.optim.Adam(groups)
    frozen = {name: value.detach() for name, value in generator.named_parameters()}
    fixed_splauts = generator(initial.latents, initial.durations).detach()
    trajectories, reconstruction_losses, objective_losses, errors = [], [], [], []
    for step in range(steps + 1):
        state = parameters()
        splauts = (torch.func.functional_call(generator, frozen, (state.latents, state.durations))
                   if optimize_latents else fixed_splauts)
        prediction = render(splauts, state.positions, state.onsets, state.amplitudes,
                            stations, times, frequencies, config)
        reconstruction = normalized_mse(prediction, observation)
        loss = reconstruction
        if position_weight:
            loss = loss + position_weight * ((state.positions - true_positions) / config.area_size).square().mean()
        if onset_weight:
            loss = loss + onset_weight * ((state.onsets - true_onsets) / config.scene_duration).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite localization objective")
        trajectories.append(state.positions.detach().clone())
        reconstruction_losses.append(float(reconstruction.detach()))
        objective_losses.append(float(loss.detach()))
        if true_positions is not None:
            errors.append((state.positions.detach() - true_positions).norm(dim=-1))
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return LocalizationResult(state.detach(), prediction.detach(), torch.stack(trajectories),
                              reconstruction_losses, objective_losses,
                              torch.stack(errors) if errors else None)


@torch.no_grad()
def spatial_loss_grid(generator: SplautGenerator, state: EventState, observation: Tensor,
                      stations: Tensor, times: Tensor, frequencies: Tensor,
                      config: Config, resolution: int = 25, event_index: int = 0) -> tuple[Tensor, Tensor]:
    """Conditional reconstruction landscape; hold all other events/parameters fixed."""
    if resolution < 2 or not 0 <= event_index < len(state.positions):
        raise ValueError("Invalid grid resolution or event index")
    validate_stations(stations)
    axis = torch.linspace(0, config.area_size, resolution)
    splauts = generator(state.latents, state.durations)
    losses = torch.empty(resolution, resolution)
    positions = state.positions.clone()
    for iy, y in enumerate(axis):
        for ix, x in enumerate(axis):
            positions[event_index] = torch.stack((x, y))
            prediction = render(splauts, positions, state.onsets, state.amplitudes,
                                stations, times, frequencies, config)
            losses[iy, ix] = normalized_mse(prediction, observation)
    return axis, losses


@torch.no_grad()
def initialize_positions(generator: SplautGenerator, initial: EventState, observation: Tensor,
                         stations: Tensor, times: Tensor, frequencies: Tensor, config: Config,
                         resolution: int = 13, sweeps: int = 1) -> EventState:
    """Coordinate grid search for a data-only start, followed by localize() refinement.

    Known event associations let us vary one position at a time. Keep the current
    position when no grid candidate improves reconstruction. No labels are used.
    """
    if sweeps < 1:
        raise ValueError("Use at least one coordinate-search sweep")
    state = initial.detach()
    splauts = generator(state.latents, state.durations)
    for _ in range(sweeps):
        for k in range(len(state.positions)):
            baseline = normalized_mse(render(splauts, state.positions, state.onsets, state.amplitudes,
                                              stations, times, frequencies, config), observation)
            axis, losses = spatial_loss_grid(generator, state, observation, stations, times,
                                             frequencies, config, resolution, k)
            flat_index = int(losses.argmin())
            iy, ix = divmod(flat_index, resolution)
            if losses[iy, ix] < baseline:
                state.positions[k] = torch.stack((axis[ix], axis[iy]))
    return state
