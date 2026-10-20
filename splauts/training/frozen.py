"""Infer events from mixtures through a frozen, supervised single-call prior."""

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from ..config import Config
from ..model.call_prior import codes_from_logits, logits_from_codes
from ..model.events import EventParameters, EventState
from ..model.generator import evaluate_field
from ..model.renderer import render
from ..propagation import absorption_transmission
from .mixtures import (MixtureFitResult, MixtureObservations, MixtureProposals,
                       _band_penalty, interpolate_power, mixture_losses, normalized_splauts)
from .pretrain import FrozenCallPrior, decoder_fingerprint


@dataclass
class FrozenMixtureFitResult(MixtureFitResult):
    selected_families: Tensor
    bank_indices: list[int]
    decoder_fingerprint: str


def interpolate_absorbed_bank(correlations: Tensor, energies: Tensor, distances: Tensor,
                              queries: Tensor, dt: float, distance_step: float) -> tuple[Tensor, Tensor]:
    """Bilinear distance/time interpolation for proposal scoring, zero beyond time grid.

    correlations [D,M,T,B], energies [D,B], distances [C,M], queries [C,M,U].
    Continuous refinement uses the exact frequency-dependent transfer function.
    """
    dc = distances / distance_step
    dl = dc.floor().long().clamp(0, len(energies)-2)
    dw = (dc - dl).clamp(0, 1)
    tc = queries / dt
    tl = tc.floor().long()
    tw = (tc - tl)[..., None]
    stations = torch.arange(correlations.shape[1])[None, :, None]
    def sample(d, t):
        valid = ((t >= 0) & (t < correlations.shape[2]))[..., None]
        return correlations[d[:, :, None], stations, t.clamp(0, correlations.shape[2]-1)] * valid
    a = sample(dl, tl) * (1-tw) + sample(dl, tl+1) * tw
    b = sample(dl+1, tl) * (1-tw) + sample(dl+1, tl+1) * tw
    aligned = a * (1-dw[:, :, None, None]) + b * dw[:, :, None, None]
    energy = energies[dl] * (1-dw[:, :, None]) + energies[dl+1] * dw[:, :, None]
    return aligned, energy


@torch.no_grad()
def _match_template(observed: MixtureObservations, residual: Tensor, bank_fields: Tensor,
                    config: Config, grid_size: int) -> tuple[Tensor, Tensor, Tensor, int, float]:
    """Find position, onset, power and a latent template from mixtures alone.

    Correlate each station with neural templates once, then continuously sample
    those correlations at geometry-dependent arrivals. Score the least-squares
    reduction in station-balanced reconstruction error. No call identity is given.
    """
    reference_squared = observed.power.square().mean((1, 2)).clamp_min(1e-20)
    length = bank_fields.shape[1]
    normalized = (residual / reference_squared[:, None, None]).permute(0, 2, 1)
    axis = (torch.arange(grid_size) + 0.5) * config.area_size / grid_size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    candidates = torch.stack((xx.flatten(), yy.flatten()), -1)
    distances = torch.cdist(candidates, observed.stations)
    gains = (distances + config.distance_offset).pow(-2)
    delays = distances / config.sound_speed
    dt = float(observed.times[1] - observed.times[0])
    padded = F.pad(normalized, (0, length-1))
    if config.absorption_db_per_km:
        distance_grid = torch.linspace(0, float(distances.max()) * 1.00001, config.absorption_distance_bins)
        correlation_bank, energy_bank = [], []
        for distance in distance_grid:
            transmission = absorption_transmission(distance, observed.frequencies, config, power=True)
            fields = bank_fields * transmission[None, None]
            correlation_bank.append(F.conv1d(padded, fields.permute(0, 2, 1).contiguous()).transpose(1, 2))
            energy_bank.append(fields.square().sum((1, 2)).clamp_min(1e-20))
        correlations, template_energy = torch.stack(correlation_bank), torch.stack(energy_bank)
    else:
        correlations = F.conv1d(padded, bank_fields.permute(0, 2, 1).contiguous()).transpose(1, 2)
        template_energy = bank_fields.square().sum((1, 2)).clamp_min(1e-20)
    best_score, selected = -math.inf, None
    for start in range(0, len(candidates), 4):
        stop = min(start + 4, len(candidates))
        query = observed.times[None, None, :] + delays[start:stop, :, None]
        if config.absorption_db_per_km:
            aligned, energy = interpolate_absorbed_bank(correlations, template_energy, distances[start:stop],
                                                        query, dt, float(distance_grid[1]))
            denominator = (energy * (gains[start:stop].square() / reference_squared[None])[:, :, None]).sum(1)
        else:
            aligned = interpolate_power(correlations, query, dt)
            denominator = ((gains[start:stop].square() / reference_squared[None]).sum(1)[:, None]
                           * template_energy[None])
        numerator = (aligned * gains[start:stop, :, None, None]).sum(1).clamp_min(0)
        score = numerator.square() / denominator[:, None, :]
        latest = observed.times[-1] - config.call_duration - delays[start:stop].max(1).values
        valid = observed.times[None, :] <= latest[:, None]
        score[~valid] = -math.inf
        values, flat = score.flatten(1).max(1)
        candidate = int(values.argmax())
        if float(values[candidate]) > best_score:
            best_score = float(values[candidate])
            time_index, bank_index = divmod(int(flat[candidate]), len(bank_fields))
            amplitude = (numerator[candidate, time_index, bank_index] / denominator[candidate, bank_index]).clamp_min(1e-8)
            selected = (candidates[start+candidate].clone(), observed.times[time_index].clone(),
                        amplitude, bank_index, best_score)
    if selected is None:
        raise ValueError("Recording is too short for the call and propagation priors")
    return selected


def _optimize(prior: FrozenCallPrior, observed: MixtureObservations, initial: EventState,
               families: Tensor, proposals: MixtureProposals, config: Config,
               steps: int, log_every: int = 0) -> MixtureFitResult:
    # Internal latent parameters are four unconstrained acoustic logits. Every
    # decoder call maps these back to a bounded acoustic code and fixed family.
    internal = initial.detach()
    internal.latents = logits_from_codes(internal.latents, config)
    parameters = EventParameters(internal, config.area_size, float(observed.times[-1]) - config.call_duration,
                                 optimize_onsets=True, optimize_amplitudes=True, optimize_latents=True)
    optimizer = torch.optim.Adam([{"params": [parameters.position_logits], "lr": 0.02},
                                  {"params": [parameters.onset_logits], "lr": 0.008},
                                  {"params": [parameters.log_amplitudes], "lr": 0.015},
                                  {"params": [parameters.latents], "lr": 0.012}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))
    positions, onsets, objectives, power_losses, compressed_losses = [], [], [], [], []
    best_loss, best_step, best_events = math.inf, 0, None
    for step in range(steps + 1):
        state = parameters()
        codes = codes_from_logits(families, state.latents, config.latent_dim, config)
        splauts = normalized_splauts(prior.generator, codes, state.durations, prior.relative_times, observed.frequencies)
        prediction = render(splauts, state.positions, state.onsets, state.amplitudes, observed.stations,
                            observed.times, observed.frequencies, config)
        power, compressed = mixture_losses(prediction, observed.power)
        # Regularize spread within a harmonic in fundamental coordinates. The
        # intended spacing between harmonics must not incur a narrow-band penalty.
        orders = (torch.arange(config.splauts) % config.harmonic_count + 1).to(splauts.frequency)
        fundamental = replace(splauts, frequency=splauts.frequency / orders,
                              frequency_width=splauts.frequency_width / orders)
        objective = 0.5 * (power + compressed) + 0.001 * _band_penalty(fundamental, 500. * config.frequency_scale)
        if not torch.isfinite(objective):
            raise RuntimeError("Non-finite frozen-prior objective")
        value = float(objective.detach())
        positions.append(state.positions.detach().clone())
        onsets.append(state.onsets.detach().clone())
        objectives.append(value)
        power_losses.append(float(power.detach()))
        compressed_losses.append(float(compressed.detach()))
        if value < best_loss:
            best_loss, best_step = value, step
            best_events = {name: tensor.detach().clone() for name, tensor in parameters.state_dict().items()}
        if log_every and (step % log_every == 0 or step == steps):
            print(f"  frozen step {step}/{steps}: objective={value:.4f}, power NMSE={float(power.detach()):.4f}", flush=True)
        if step < steps:
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters.parameters(), 10.)
            optimizer.step()
            scheduler.step()
    parameters.load_state_dict(best_events)
    with torch.no_grad():
        state = parameters().detach()
        state.latents = codes_from_logits(families, state.latents, config.latent_dim, config)
        splauts = normalized_splauts(prior.generator, state.latents, state.durations,
                                     prior.relative_times, observed.frequencies).detach()
        prediction = render(splauts, state.positions, state.onsets, state.amplitudes, observed.stations,
                            observed.times, observed.frequencies, config)
    if best_step != steps:
        positions.append(state.positions)
        onsets.append(state.onsets)
    return MixtureFitResult(prior.generator, state, splauts, prediction, proposals, torch.stack(positions),
                            torch.stack(onsets), objectives, power_losses, compressed_losses, [], best_step)


def fit_frozen_prior(observed: MixtureObservations, prior: FrozenCallPrior, config: Config,
                      event_count: int, steps: int = 400, proposal_steps: int = 80,
                      proposal_grid_size: int = 13, log_every: int = 100) -> FrozenMixtureFitResult:
    """Infer unknown held-out calls, without changing the pretrained decoder.

    Neural matching pursuit chooses call family and an initial acoustic code.
    Within each selected family, source position, onset, power and acoustic
    coordinates are differentiable. Neither held-out calls nor their embeddings,
    identities, locations, onsets or amplitudes are supplied.
    """
    observed.validate()
    if not 1 <= event_count <= 5 or min(steps, proposal_steps) < 0 or proposal_grid_size < 2:
        raise ValueError("Use 1–5 events, nonnegative iterations and grid size >=2")
    if decoder_fingerprint(prior.generator) != prior.fingerprint:
        raise ValueError("Pretrained decoder changed after its checkpoint was frozen")
    if any(parameter.requires_grad for parameter in prior.generator.parameters()):
        raise ValueError("Decoder parameters must be frozen before mixture inversion")
    keys = ("sample_rate", "n_fft", "hop_length", "call_duration", "splauts", "latent_dim",
            "harmonic_count", "harmonic_rolloff", "frequency_scale", "call_families")
    if any(getattr(config, key) != getattr(prior.generator.config, key) for key in keys):
        raise ValueError("Mixture configuration differs from the pretrained decoder")
    if observed.frequencies.shape != prior.frequencies.shape or not torch.allclose(observed.frequencies, prior.frequencies):
        raise ValueError("Observation frequencies must match the pretrained STFT grid")
    if not torch.allclose(observed.times[1] - observed.times[0],
                           prior.relative_times[1] - prior.relative_times[0], atol=1e-6, rtol=1e-4):
        raise ValueError("Observation time step must match the pretrained STFT grid")
    with torch.no_grad():
        durations = torch.full((len(prior.bank_codes),), config.call_duration)
        bank_fields = evaluate_field(normalized_splauts(prior.generator, prior.bank_codes, durations,
                                                        prior.relative_times, observed.frequencies),
                                     prior.relative_times, observed.frequencies)
    states, fields, scores, bank_indices, family_values = [], [], [], [], []
    total = torch.zeros_like(observed.power)
    for k in range(event_count):
        residual = (observed.power - total).clamp_min(0)
        with torch.no_grad():
            position, onset, amplitude, bank_index, score = _match_template(observed, residual, bank_fields,
                                                                           config, proposal_grid_size)
            codes = prior.bank_codes[bank_index:bank_index+1].clone()
            family = codes[:, :len(config.call_families)].argmax(-1)
            state = EventState(position[None], onset[None], amplitude[None], torch.tensor([config.call_duration]), codes)
            single_proposals = MixtureProposals(state.positions, state.onsets,
                                                 bank_fields[bank_index:bank_index+1] * amplitude,
                                                 prior.relative_times, [score])
        residual_observations = MixtureObservations(residual, observed.stations, observed.times, observed.frequencies)
        single = _optimize(prior, residual_observations, state, family, single_proposals, config, proposal_steps)
        states.append(single.state)
        with torch.no_grad():
            fields.append(evaluate_field(single.splauts, prior.relative_times, observed.frequencies)[0] * single.state.amplitudes[0])
        total += single.prediction
        scores.append(score)
        bank_indices.append(bank_index)
        family_values.append(family[0])
        if log_every:
            print(f"  proposed event {k+1}/{event_count}: family={int(family[0])}, position={single.state.positions[0].tolist()}", flush=True)
    initial = EventState(**{name: torch.cat([getattr(state, name) for state in states])
                           for name in ("positions", "onsets", "amplitudes", "durations", "latents")})
    proposals = MixtureProposals(initial.positions, initial.onsets, torch.stack(fields), prior.relative_times, scores)
    families = torch.stack(family_values)
    result = _optimize(prior, observed, initial, families, proposals, config, steps, log_every)
    if decoder_fingerprint(prior.generator) != prior.fingerprint:
        raise RuntimeError("Frozen decoder weights changed during inference")
    if any(parameter.grad is not None for parameter in prior.generator.parameters()):
        raise RuntimeError("Frozen decoder accumulated gradients")
    return FrozenMixtureFitResult(**vars(result), selected_families=families, bank_indices=bank_indices,
                                  decoder_fingerprint=prior.fingerprint)
