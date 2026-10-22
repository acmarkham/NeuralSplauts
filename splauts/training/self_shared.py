"""Conservative sensor-self / propagated-shared decomposition of power fields.

Self is a station-local, zero-delay residual defined by conservative energy
transfer as relu(Y - shared). It cannot reduce its penalty by discarding data,
or hide shared overprediction with negative energy. Shared gates start at zero.
"""

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import Config
from ..model.events import EventParameters, EventState
from ..model.generator import Splauts, SplautGenerator, evaluate_field
from ..model.renderer import render
from .mixtures import (MixtureObservations, MixtureProposals, interpolate_power,
                       normalized_splauts)


@dataclass(frozen=True)
class SelfSharedLoss:
    reconstruction: float = 10.
    self_energy: float = 1.
    spatial_sparsity: float = .08
    multisensor: float = 4.
    gate_scale: float = .005
    min_stations: float = 4.
    correlation_threshold: float = .15
    correlation_temperature: float = .03
    spatial_radius_m: float = 15.

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in
               (self.reconstruction, self.self_energy, self.spatial_sparsity, self.multisensor)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if (not 0 < self.gate_scale or not 1 <= self.min_stations
                or not 0 <= self.correlation_threshold < 1 or self.correlation_temperature <= 0
                or not math.isfinite(self.spatial_radius_m) or self.spatial_radius_m <= 0):
            raise ValueError("Use positive gate scale and at least one supporting station")


@dataclass
class SelfSharedFit:
    generator: SplautGenerator
    state: EventState
    splauts: Splauts
    shared: Tensor
    self_power: Tensor
    gates: Tensor
    support: Tensor
    proposals: MixtureProposals
    history: list[dict]
    best_step: int


def effective_stations(station_mass: Tensor) -> Tensor:
    """Inverse concentration, independent of overall source scale (1 to M)."""
    normalized = station_mass / station_mass.sum(-1, keepdim=True).clamp_min(1e-20)
    inverse = 1 / normalized.square().sum(-1).clamp_min(1e-20)
    return torch.where(station_mass.sum(-1) > 1e-20, inverse, torch.zeros_like(inverse))


def decomposition_loss(components: Tensor, observed: Tensor, gates: Tensor,
                       support: Tensor, weights: SelfSharedLoss, positions: Tensor | None = None,
                       time_mask: Tensor | None = None):
    """Joint objective; residual self has no propagation and no learned capacity limit.

    Gates are received-power mass in fixed, station-balanced observation units.
    A saturating spatial penalty counts occupied locations rather than Gaussian
    pixels; nearby groups pool their mass, and every Gaussian within one group
    has the same source location and onset.
    """
    shared = components.sum(0)
    local = (observed - shared).clamp_min(0)
    full_local = local
    if time_mask is not None:
        shared, local, observed = shared[:, time_mask], local[:, time_mask], observed[:, time_mask]
    mu = observed.mean((1, 2)).clamp_min(1e-12)
    rms2 = observed.square().mean((1, 2)).clamp_min(1e-20)
    reconstruction = ((local + shared - observed).square().mean((1, 2)) / rms2).mean()
    self_energy = (local.mean((1, 2)) / mu).mean()
    if positions is None:
        neighborhood_mass = gates
    else:
        affinity = torch.exp(-torch.cdist(positions,positions).square()/(2*weights.spatial_radius_m**2))
        neighborhood_mass = affinity @ gates
    # Coincident groups pay once based on their aggregate mass. Splitting one
    # source into several spectral/time groups must not invent extra locations.
    spatial = (gates / (neighborhood_mass + weights.gate_scale)).sum()
    support_penalty = (gates * F.relu(1 - support / weights.min_stations)).sum()
    objective = (weights.reconstruction * reconstruction + weights.self_energy * self_energy
                 + weights.spatial_sparsity * spatial + weights.multisensor * support_penalty)
    return objective, full_local, {
        "reconstruction": reconstruction, "self_energy": self_energy,
        "spatial_sparsity": spatial, "unsupported_mass": support_penalty,
    }


def coherent_stations(unit: Tensor, observation: Tensor, weights: SelfSharedLoss) -> Tensor:
    """Count sensors agreeing with a source's *time-varying* spectral template.

    Per-frequency temporal centering removes a shared stationary noise spectrum
    as evidence. Template spectral weights focus the test on its frequency band.
    Sensor gains cancel in the normalized correlation; no gains are fitted.
    """
    unit = unit / unit.amax((2, 3), keepdim=True).clamp_min(1e-20)
    target = observation / observation.amax((1, 2), keepdim=True).clamp_min(1e-20)
    spectral = unit.mean(2, keepdim=True)
    spectral = spectral / spectral.sum(-1, keepdim=True).clamp_min(1e-20)
    centered = unit - unit.mean(2, keepdim=True)
    target = target - target.mean(1, keepdim=True)
    numerator = (centered * target[None] * spectral).sum((2, 3))
    denominator = ((centered.square()*spectral).sum((2, 3))
                   * (target[None].square()*spectral).sum((2, 3))).clamp_min(1e-20).sqrt()
    correlation = numerator / denominator
    return torch.sigmoid((correlation-weights.correlation_threshold)/weights.correlation_temperature).sum(1)


def received_components(splauts, state, observed, config, gates, weights=SelfSharedLoss(), time_mask=None):
    """Fix the source-scale gauge by normalizing its rendered observation footprint."""
    unit = []
    for k in range(len(gates)):
        one = Splauts(**{key: getattr(splauts, key)[k:k+1] for key in vars(splauts)})
        unit.append(render(one, state.positions[k:k+1], state.onsets[k:k+1],
                           torch.ones(1), observed.stations, observed.times,
                           observed.frequencies, config))
    unit = torch.stack(unit)
    target = observed.power if time_mask is None else observed.power[:, time_mask]
    scored_unit = unit if time_mask is None else unit[:, :, time_mask]
    mu = target.mean((1, 2)).clamp_min(1e-12)
    station_mass = scored_unit.mean((2, 3)) / mu[None]
    normalizer = station_mass.mean(1).clamp_min(1e-12)
    support = torch.minimum(effective_stations(station_mass), coherent_stations(scored_unit, target, weights))
    amplitudes = gates / normalizer
    return unit * amplitudes[:, None, None, None], support, amplitudes


@torch.no_grad()
def birth_candidate(generator, parameters, gates, observed, config, proposals, weights,
                    geometry_fixed=None, time_mask=None):
    """Cross the sparse activation barrier only for a profitable mixture-only birth.

    A nearly discrete location cost has a large derivative at zero. Explicitly
    testing nonzero gates lets a useful new group appear without relaxing that
    cost or starting the experiment with pre-existing shared energy.
    """
    state = parameters()
    splauts = geometry_fixed if geometry_fixed is not None else normalized_splauts(
        generator, state.latents, state.durations, proposals.relative_times, observed.frequencies)
    unit, support, _ = received_components(splauts, state, observed, config, torch.ones_like(gates), weights, time_mask)
    current = gates.detach().clone()
    best = float(decomposition_loss(unit*current[:,None,None,None], observed.power, current, support, weights, state.positions, time_mask)[0])
    selected = None
    for k in range(len(current)):
        if float(current[k]) > 1e-6:
            continue
        for mass in (.025,.05,.1,.2,.35,.5):
            trial = current.clone()
            trial[k] = mass
            value = float(decomposition_loss(unit*trial[:,None,None,None], observed.power, trial, support, weights, state.positions, time_mask)[0])
            if value < best-1e-4:
                best, selected = value, (k, mass)
    if selected is not None:
        gates[selected[0]] = selected[1]
    return selected


@torch.no_grad()
def propose_shared(observed: MixtureObservations, config: Config, capacity=5,
                   grid_size=13, min_stations=3., time_mask=None) -> MixtureProposals:
    """Robust common-field proposals, using mixtures only.

    A lower quartile of gain-corrected aligned station powers prevents one loud
    local sensor transient from supplying the candidate's source field. Score
    and subtraction use continuous delays; final fits use the Gaussian renderer.
    """
    observed.validate()
    if time_mask is not None and (time_mask.dtype != torch.bool or time_mask.shape != observed.times.shape
                                  or int(time_mask.sum()) < 3 or time_mask.device.type != "cpu"):
        raise ValueError("time_mask must select at least three CPU arrival frames")
    if not 1 <= capacity <= 8 or grid_size < 2:
        raise ValueError("Use 1–8 shared groups and a grid >=2")
    dt = float(observed.times[1] - observed.times[0])
    relative = torch.arange(round(config.call_duration / dt) + 1) * dt
    axis = (torch.arange(grid_size) + .5) * config.area_size / grid_size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    candidates = torch.stack((xx.flatten(), yy.flatten()), -1)
    distances = torch.cdist(candidates, observed.stations)
    gains = (distances + config.distance_offset).pow(-config.power_spreading_exponent)
    delays = distances / config.sound_speed
    target = observed.power if time_mask is None else observed.power[:, time_mask]
    reference = target.square().mean((1, 2)).sqrt().clamp_min(1e-12)
    support = effective_stations(gains / reference[None])
    allowed = support >= min_stations
    if not allowed.any():
        raise ValueError("No grid point has the required multi-station footprint")
    scoring_times = observed.times[::2]
    valid = ((scoring_times >= config.call_duration/2)
             & (scoring_times <= observed.times[-1] - config.call_duration/2))
    kernel = 2 * max(1, round(config.call_duration / (4*dt))) + 1
    # Only proposal construction removes a stationary median; the fitted self
    # field and all losses still use the complete original observation.
    residual = (observed.power-target.quantile(.5, dim=1, keepdim=True)).clamp_min(0)
    positions, onsets, fields, scores = [], [], [], []
    for _ in range(capacity):
        best, best_i, best_t = -math.inf, 0, 0
        scoring_power = residual[:, ::2]
        if time_mask is not None:
            scoring_power = scoring_power * time_mask[None, ::2, None]
        for first in range(0, len(candidates), 8):
            last = min(first+8, len(candidates))
            aligned = interpolate_power(scoring_power,
                scoring_times[None, None] + delays[first:last, :, None], 2*dt)
            common = torch.quantile(aligned / gains[first:last, :, None, None], .25, dim=1)
            balanced = common[:, None] * (gains[first:last] / reference[None])[:, :, None, None]
            energy = balanced.square().mean((1, 3))
            score = F.avg_pool1d(energy[:, None], kernel, stride=1, padding=kernel//2)[:, 0]
            score[:, ~valid] = -1
            score[~allowed[first:last]] = -1
            value, ti = score.max(1)
            selected = int(value.argmax())
            if float(value[selected]) > best:
                best, best_i, best_t = float(value[selected]), first+selected, int(ti[selected])
        onset = (scoring_times[best_t] - config.call_duration/2).clamp(0, float(observed.times[-1])-config.call_duration)
        query = onset + relative[None, None] + delays[best_i][None, :, None]
        aligned = interpolate_power(residual, query, dt)[0]
        field = torch.quantile(aligned / gains[best_i, :, None, None], .25, dim=0)
        projection = interpolate_power(field[None].expand(len(observed.stations), -1, -1),
            observed.times[None, None] - onset - delays[best_i][None, :, None], dt)[0]
        residual = (residual - projection*gains[best_i, :, None, None]).clamp_min(0)
        positions.append(candidates[best_i])
        onsets.append(onset)
        fields.append(field)
        scores.append(best)
    return MixtureProposals(torch.stack(positions), torch.stack(onsets), torch.stack(fields), relative, scores)


def fit_self_shared(observed: MixtureObservations, config: Config, capacity=5,
                     weights=SelfSharedLoss(), warmup_steps=150, steps=600,
                     geometry_steps=75, grid_size=13, seed=101, log_every=100, time_mask=None,
                     spatial_exploration_steps=0):
    """Fit only station mixtures; capacity is an upper bound, not a known call count."""
    if min(warmup_steps, steps, geometry_steps, spatial_exploration_steps) < 0 or weights.min_stations > len(observed.stations):
        raise ValueError("Invalid iteration count or supporting-station count")
    proposals = propose_shared(observed, config, capacity, grid_size, weights.min_stations, time_mask)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = SplautGenerator(config)
        latents = nn.Parameter(torch.randn(capacity, config.latent_dim)*.5)
    scales = proposals.fields.square().mean((1, 2)).sqrt().clamp_min(1e-12)
    targets = proposals.fields / scales[:, None, None]
    generator.initialize_from_power(targets, proposals.relative_times, observed.frequencies,
                                    config.call_duration, seed)
    durations = torch.full((capacity,), config.call_duration)
    optimizer = torch.optim.Adam([*generator.parameters(), latents], lr=.004)
    for step in range(warmup_steps):
        splauts = normalized_splauts(generator, latents, durations, proposals.relative_times, observed.frequencies)
        field = evaluate_field(splauts, proposals.relative_times, observed.frequencies)
        loss = (field-targets).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    initial = EventState(proposals.positions, proposals.onsets, torch.ones(capacity), durations, latents.detach())
    parameters = EventParameters(initial, config.area_size, float(observed.times[-1])-config.call_duration,
                                 optimize_onsets=True, optimize_latents=True)
    gates = nn.Parameter(torch.zeros(capacity))
    optimizer = torch.optim.Adam([
        {"params":[gates], "lr":.008},
        {"params":[parameters.position_logits], "lr":.015},
        {"params":[parameters.onset_logits], "lr":.005},
        {"params":[parameters.latents, *generator.parameters()], "lr":.0015},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))
    fixed = normalized_splauts(generator, initial.latents, durations,
                               proposals.relative_times, observed.frequencies).detach()
    history, best, best_step, selected = [], math.inf, 0, None
    for step in range(steps+1):
        state = parameters()
        splauts = fixed if step < geometry_steps else normalized_splauts(
            generator, state.latents, durations, proposals.relative_times, observed.frequencies)
        active_gates = gates.clamp_min(0)
        components, support, amplitudes = received_components(splauts, state, observed, config, active_gates, weights, time_mask)
        loss, local, terms = decomposition_loss(components, observed.power, active_gates, support, weights, state.positions, time_mask)
        # Continuation lets zero gates cross a sparse activation barrier while
        # geometry and templates improve. Checkpoint selection ALWAYS uses the
        # same final objective above, including the initial all-self model.
        ramp = min(1., step/max(1., steps/2))
        if spatial_exploration_steps:
            ramp = max(0., min(1., (step-spatial_exploration_steps)/max(1., .75*(steps-spatial_exploration_steps))))
        optimization_weights = replace(weights,
            spatial_sparsity=weights.spatial_sparsity*(ramp if spatial_exploration_steps else .25+.75*ramp),
            gate_scale=(1-ramp)*max(.03, weights.gate_scale)+ramp*weights.gate_scale,
            min_stations=(1-ramp)*min(3.,weights.min_stations)+ramp*weights.min_stations,
            multisensor=weights.multisensor*(.5+.5*ramp))
        optimization_loss = decomposition_loss(components, observed.power, active_gates, support, optimization_weights, state.positions, time_mask)[0]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite self/shared objective")
        value = float(loss.detach())
        history.append({"step":step, "objective":value,
                        "optimization_objective":float(optimization_loss.detach()), "continuation_fraction":ramp,
                        **{k:float(v.detach()) for k,v in terms.items()},
                        "gate_mass":active_gates.detach().tolist(), "effective_stations":support.detach().tolist()})
        if value < best:
            best, best_step = value, step
            selected = ({k:v.detach().clone() for k,v in generator.state_dict().items()},
                        {k:v.detach().clone() for k,v in parameters.state_dict().items()}, gates.detach().clone(), splauts.detach())
        if log_every and (step % log_every == 0 or step == steps):
            print(f"  self/shared {step}/{steps}: loss={value:.4f}, self={float(terms['self_energy'].detach()):.3f}, gates={active_gates.detach().tolist()}", flush=True)
        # If no gate has ever opened, geometry and shape have received zero
        # gradients. Once continuation reaches the fixed final objective,
        # repeating the same births and gradients cannot change the result.
        if ramp == 1 and step >= geometry_steps+1 and all(
                not any(h["gate_mass"]) for h in history):
            if log_every:
                print("  converged at all-self after full continuation", flush=True)
            break
        if step < steps:
            optimizer.zero_grad()
            optimization_loss.backward()
            torch.nn.utils.clip_grad_norm_([*generator.parameters(), *parameters.parameters(), gates], 10.)
            optimizer.step()
            with torch.no_grad():
                gates.clamp_(min=0)
            if step % 40 == 0:
                born = birth_candidate(generator, parameters, gates, observed, config,
                                       proposals, optimization_weights, fixed if step < geometry_steps else None, time_mask)
                if born is not None:
                    history[-1]["birth_after_step"] = {"group":born[0], "mass":born[1]}
            scheduler.step()
    generator.load_state_dict(selected[0])
    parameters.load_state_dict(selected[1])
    with torch.no_grad():
        state = parameters().detach()
        splauts = selected[3]
        components, support, amplitudes = received_components(splauts, state, observed, config, selected[2], weights, time_mask)
        state.amplitudes = amplitudes
        local = (observed.power-components.sum(0)).clamp_min(0)
    return SelfSharedFit(generator, state, splauts, components.sum(0), local,
                         selected[2], support, proposals, history, best_step)
