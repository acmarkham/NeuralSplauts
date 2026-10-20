"""Mixture-only scene fitting: no source audio, call specifications or labels.

Known priors: event count, common duration, station geometry and propagation.
Candidates and provisional fields come from delayed station mixtures. A fresh
shared decoder first approximates those provisional fields, then is optimized
through multistation reconstruction. Each generated field has unit RMS on the
duration/frequency grid; a learned event power scale removes the otherwise
redundant amplitude/decoder gauge and handles weak and strong candidates alike.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import Config, validate_stations
from ..model.events import EventParameters, EventState
from ..model.generator import SplautGenerator, Splauts, evaluate_field
from ..model.renderer import render


@dataclass(frozen=True)
class MixtureObservations:
    power: Tensor
    stations: Tensor
    times: Tensor
    frequencies: Tensor

    def validate(self) -> None:
        validate_stations(self.stations)
        if self.power.shape != (len(self.stations), len(self.times), len(self.frequencies)):
            raise ValueError("Power must have shape [stations, time, frequency]")
        if len(self.times) < 3 or len(self.frequencies) < 3:
            raise ValueError("At least three time and frequency samples are required")
        for value in (self.power, self.stations, self.times, self.frequencies):
            if value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError("Mixture observations must be finite CPU tensors")
        if (self.power < 0).any() or (self.power.square().mean((1, 2)) <= 0).any():
            raise ValueError("Each station needs nonnegative power and positive signal energy")
        delta = self.times.diff()
        if (delta <= 0).any() or not torch.allclose(delta, delta[:1].expand_as(delta), atol=1e-6, rtol=1e-4):
            raise ValueError("Time samples must be increasing and uniformly spaced")
        if (self.frequencies.diff() <= 0).any() or float(self.times[0]) != 0:
            raise ValueError("Frequencies must increase and recording times must start at zero")


@dataclass
class MixtureProposals:
    positions: Tensor
    onsets: Tensor
    fields: Tensor  # Provisional emitted powers derived ONLY from mixtures.
    relative_times: Tensor
    scores: list[float]


@dataclass
class MixtureFitResult:
    generator: SplautGenerator
    state: EventState
    splauts: Splauts
    prediction: Tensor
    proposals: MixtureProposals
    trajectories: Tensor
    onset_history: Tensor
    objectives: list[float]
    power_losses: list[float]
    compressed_losses: list[float]
    warmup_losses: list[float]
    best_step: int


def interpolate_power(power: Tensor, queries: Tensor, step: float) -> Tensor:
    """Linear interpolation of [M,T,F] at [candidates,M,U] continuous times.

    This is only for initializing candidates from observed grids. Final fitting
    uses the continuous Gaussian renderer rather than interpolated templates.
    """
    coordinates = queries / step
    left = coordinates.floor().long()
    fraction = (coordinates - left)[..., None]
    expanded = power[None].expand(len(queries), -1, -1, -1)
    def sample(indices: Tensor) -> Tensor:
        valid = ((indices >= 0) & (indices < power.shape[1]))[..., None]
        indices = indices.clamp(0, power.shape[1] - 1)[..., None].expand(-1, -1, -1, power.shape[2])
        return torch.gather(expanded, 2, indices) * valid
    return sample(left) * (1 - fraction) + sample(left + 1) * fraction


def mixture_losses(prediction: Tensor, observation: Tensor) -> tuple[Tensor, Tensor]:
    """Station-balanced power and square-root-power errors, both mixture-only."""
    power = ((prediction - observation).square().mean((1, 2))
             / observation.square().mean((1, 2)).clamp_min(1e-20)).mean()
    floor = observation.amax((1, 2), keepdim=True) * 1e-6 + 1e-12
    compressed = ((torch.sqrt(prediction.clamp_min(0) + floor) - torch.sqrt(observation + floor))
                  .square().mean((1, 2)) / observation.mean((1, 2)).clamp_min(1e-20)).mean()
    return power, compressed


@torch.no_grad()
def propose_events(observed: MixtureObservations, config: Config, event_count: int,
                   grid_size: int = 13, bandwidth_hz: float = 500.0) -> MixtureProposals:
    """Greedy propagation-aware matching pursuit on the station mixtures.

    For each candidate location, undo travel delays and find a nonnegative
    least-squares common emitted field. A duration/frequency window chooses a
    call-sized patch maximizing explained, station-balanced mixture energy.
    Subtract its projected field before proposing the next event.
    """
    observed.validate()
    if not 1 <= event_count <= 5 or grid_size < 2 or bandwidth_hz <= 0:
        raise ValueError("Use 1–5 events, a grid of at least 2 and a positive bandwidth")
    if float(observed.times[-1]) <= config.call_duration:
        raise ValueError("Recording must be longer than the duration prior")
    dt = float(observed.times[1] - observed.times[0])
    relative = torch.arange(round(config.call_duration / dt) + 1, dtype=observed.power.dtype) * dt
    # Cell centres avoid singularly high gains and saturated coordinate logits
    # when a residual happens to favour a candidate exactly on a station.
    axis = (torch.arange(grid_size, dtype=observed.power.dtype) + 0.5) * config.area_size / grid_size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    candidates = torch.stack((xx.flatten(), yy.flatten()), -1)
    distances = torch.cdist(candidates, observed.stations)
    gains = (distances + config.distance_offset).pow(-2)
    delays = distances / config.sound_speed
    reference = observed.power.square().mean((1, 2)).sqrt().clamp_min(1e-10)
    residual = observed.power.clone()
    positions, onsets, fields, scores = [], [], [], []
    # Decimation is for proposal scoring only; provisional fields and final fits
    # use every original STFT bin and frame.
    scoring_times, scoring_frequencies = observed.times[::2], observed.frequencies[::2]
    time_kernel = 2 * max(1, round(config.call_duration / (2 * dt * 2))) + 1
    df = float(scoring_frequencies[1] - scoring_frequencies[0])
    frequency_kernel = 2 * max(1, round(bandwidth_hz / (2 * df))) + 1
    valid_times = ((scoring_times >= config.call_duration / 2)
                   & (scoring_times <= observed.times[-1] - config.call_duration / 2))
    for _ in range(event_count):
        scoring_power = residual[:, ::2, ::2] / reference[:, None, None]
        best_score, best_index, best_t, best_f = -math.inf, 0, 0, 0
        for start in range(0, len(candidates), 8):
            stop = min(start + 8, len(candidates))
            queries = scoring_times[None, None, :] + delays[start:stop, :, None]
            aligned = interpolate_power(scoring_power, queries, dt * 2)
            unit_gains = gains[start:stop] / reference[None, :]
            unit_gains = F.normalize(unit_gains, dim=1)
            projected = (aligned * unit_gains[:, :, None, None]).sum(1)
            score = F.avg_pool2d(projected.square()[:, None], (time_kernel, 1), stride=1,
                                 padding=(time_kernel // 2, 0))
            score = F.avg_pool2d(score, (1, frequency_kernel), stride=1,
                                 padding=(0, frequency_kernel // 2))[:, 0]
            score[:, ~valid_times] = -1
            value, flat = score.flatten(1).max(1)
            selected = int(value.argmax())
            if float(value[selected]) > best_score:
                best_score = float(value[selected])
                best_index = start + selected
                best_t, best_f = divmod(int(flat[selected]), len(scoring_frequencies))
        position = candidates[best_index]
        onset = (scoring_times[best_t] - config.call_duration / 2).clamp(0, float(observed.times[-1]) - config.call_duration)
        query = onset + relative[None, None, :] + delays[best_index][None, :, None]
        aligned = interpolate_power(residual, query, dt)[0]
        weights = gains[best_index] / reference.square()
        emitted = (aligned * weights[:, None, None]).sum(0) / (gains[best_index] * weights).sum().clamp_min(1e-20)
        centre = scoring_frequencies[best_f]
        mask = (observed.frequencies - centre).abs() <= bandwidth_hz / 2
        # Re-centre a flat box-score maximum on the energy within its band.
        energy = emitted.sum(0) * mask
        centre = (energy * observed.frequencies).sum() / energy.sum().clamp_min(1e-20)
        mask = (observed.frequencies - centre).abs() <= bandwidth_hz / 2
        emitted = emitted * mask[None]
        projected_times = observed.times[None, None, :] - onset - delays[best_index][None, :, None]
        projected = interpolate_power(emitted[None].expand(len(observed.stations), -1, -1), projected_times, dt)[0]
        residual = (residual - projected * gains[best_index, :, None, None]).clamp_min(0)
        positions.append(position.clone())
        onsets.append(onset)
        fields.append(emitted)
        scores.append(best_score)
    return MixtureProposals(torch.stack(positions), torch.stack(onsets), torch.stack(fields), relative, scores)


def _band_penalty(splauts: Splauts, bandwidth_hz: float) -> Tensor:
    mass = splauts.weight * splauts.time_width * splauts.frequency_width
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
    centre = (mass * splauts.frequency).sum(-1, keepdim=True)
    variance = (mass * ((splauts.frequency - centre).square() + splauts.frequency_width.square())).sum(-1)
    return (variance / bandwidth_hz ** 2).mean()


def normalized_splauts(generator: SplautGenerator, latents: Tensor, durations: Tensor,
                        relative_times: Tensor, frequencies: Tensor) -> Splauts:
    """Fix the field-power gauge using unit RMS on a shared event-relative grid."""
    splauts = generator(latents, durations)
    field = evaluate_field(splauts, relative_times, frequencies)
    rms = field.square().mean((1, 2)).sqrt().clamp_min(1e-8)
    return Splauts(splauts.time, splauts.frequency, splauts.time_width,
                    splauts.frequency_width, splauts.weight / rms[:, None])


def fit_mixtures(observed: MixtureObservations, config: Config, event_count: int,
                 warmup_steps: int = 250, steps: int = 400, geometry_steps: int = 75,
                 proposal_grid_size: int = 13, fit_seed: int = 101,
                 bandwidth_hz: float = 500.0, log_every: int = 100) -> MixtureFitResult:
    """Fit from mixtures alone; no pretrained decoder or source-label argument.

    Warmup targets are provisional back-projections of mixtures, not isolated
    calls. The final objective uses only observed station mixtures plus a small
    spectral-extent prior. Model selection also uses this objective only.
    """
    if min(warmup_steps, steps, geometry_steps) < 0:
        raise ValueError("Iteration counts must be nonnegative")
    proposals = propose_events(observed, config, event_count, proposal_grid_size, bandwidth_hz)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fit_seed)
        generator = SplautGenerator(config)
        latents = nn.Parameter(torch.randn(event_count, config.latent_dim) * 0.5)
    source_scales = proposals.fields.square().mean((1, 2)).sqrt().clamp_min(1e-8)
    shape_targets = proposals.fields / source_scales[:, None, None]
    generator.initialize_from_power(shape_targets, proposals.relative_times, observed.frequencies,
                                    config.call_duration, fit_seed)
    durations = torch.full((event_count,), config.call_duration)
    warm_optimizer = torch.optim.Adam([*generator.parameters(), latents], lr=0.004)
    warmup_losses = []
    for step in range(warmup_steps + 1):
        splauts = normalized_splauts(generator, latents, durations, proposals.relative_times, observed.frequencies)
        field = evaluate_field(splauts, proposals.relative_times, observed.frequencies)
        loss = (field - shape_targets).square().mean()
        warmup_losses.append(float(loss.detach()))
        if step < warmup_steps:
            warm_optimizer.zero_grad()
            loss.backward()
            warm_optimizer.step()
    initial = EventState(proposals.positions, proposals.onsets, source_scales, durations, latents.detach())
    parameters = EventParameters(initial, config.area_size, float(observed.times[-1]) - config.call_duration,
                                 optimize_onsets=True, optimize_amplitudes=True, optimize_latents=True)
    optimizer = torch.optim.Adam([{"params": [parameters.position_logits], "lr": 0.02},
                                  {"params": [parameters.onset_logits], "lr": 0.006},
                                  {"params": [parameters.log_amplitudes], "lr": 0.012},
                                  {"params": [parameters.latents], "lr": 0.004},
                                  {"params": list(generator.parameters()), "lr": 0.002}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    fixed = normalized_splauts(generator, initial.latents, durations, proposals.relative_times, observed.frequencies).detach()
    positions, onsets, objectives, power_losses, compressed_losses = [], [], [], [], []
    best_loss, best_step, best_generator, best_events = math.inf, 0, None, None
    for step in range(steps + 1):
        state = parameters()
        splauts = (fixed if step < geometry_steps else normalized_splauts(generator, state.latents, state.durations,
                                                                         proposals.relative_times, observed.frequencies))
        prediction = render(splauts, state.positions, state.onsets, state.amplitudes, observed.stations,
                            observed.times, observed.frequencies, config)
        power, compressed = mixture_losses(prediction, observed.power)
        objective = 0.5 * (power + compressed) + 0.001 * _band_penalty(splauts, bandwidth_hz)
        if not torch.isfinite(objective):
            raise RuntimeError("Non-finite mixture fitting objective")
        value = float(objective.detach())
        positions.append(state.positions.detach().clone())
        onsets.append(state.onsets.detach().clone())
        objectives.append(value)
        power_losses.append(float(power.detach()))
        compressed_losses.append(float(compressed.detach()))
        if value < best_loss:
            best_loss, best_step = value, step
            best_generator = {name: tensor.detach().clone() for name, tensor in generator.state_dict().items()}
            best_events = {name: tensor.detach().clone() for name, tensor in parameters.state_dict().items()}
        if log_every and (step % log_every == 0 or step == steps):
            print(f"  mixture step {step}/{steps}: objective={value:.4f}, power NMSE={float(power.detach()):.4f}", flush=True)
        if step < steps:
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_([*generator.parameters(), *parameters.parameters()], 10.)
            optimizer.step()
            scheduler.step()
    generator.load_state_dict(best_generator)
    parameters.load_state_dict(best_events)
    generator.eval()
    with torch.no_grad():
        state = parameters().detach()
        splauts = normalized_splauts(generator, state.latents, state.durations,
                                     proposals.relative_times, observed.frequencies).detach()
        prediction = render(splauts, state.positions, state.onsets, state.amplitudes, observed.stations,
                            observed.times, observed.frequencies, config)
    # Preserve the selected endpoint as well as the entire optimization trace.
    if best_step != steps:
        positions.append(state.positions)
        onsets.append(state.onsets)
    return MixtureFitResult(generator, state, splauts, prediction, proposals,
                            torch.stack(positions), torch.stack(onsets), objectives, power_losses,
                            compressed_losses, warmup_losses, best_step)
