"""Supervise a shared call prior on isolated calls, then freeze it for inversion."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor

from ..config import Config
from ..model.call_prior import acoustic_ranges, CallPriorGenerator
from ..model.generator import evaluate_field
from ..simulator.calls import CallSpec, instantaneous_frequency, power_spectrogram, synthesize_call
from .mixtures import normalized_splauts


@dataclass
class FrozenCallPrior:
    generator: CallPriorGenerator
    bank_codes: Tensor
    relative_times: Tensor
    frequencies: Tensor
    fingerprint: str


@dataclass
class PriorTrainingResult:
    prior: FrozenCallPrior
    calls: list[CallSpec]
    codes: Tensor
    targets: Tensor
    training_history: list[dict]
    train_nmse: list[float]
    validation_nmse: list[float]
    validation_calls: list[CallSpec]


def decoder_fingerprint(generator: CallPriorGenerator) -> str:
    digest = hashlib.sha256()
    for name, value in generator.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sample_training_calls(config: Config, per_family: int, seed: int) -> list[CallSpec]:
    if per_family < 1:
        raise ValueError("Supply at least one isolated example per family")
    rng = torch.Generator().manual_seed(seed)
    calls = []
    for kind in config.call_families:
        for values in torch.rand(per_family, 4, generator=rng):
            f, b, m, p = [lo + float(u) * (hi-lo) for u, (lo, hi) in zip(values, acoustic_ranges(config))]
            calls.append(CallSpec(kind, config.call_duration, f, b, m, p,
                                  config.harmonic_count, config.harmonic_rolloff))
    return calls


def encode_calls(calls: list[CallSpec], latent_dim: int, config: Config = Config()) -> Tensor:
    start = len(config.call_families)
    if latent_dim < start + 4:
        raise ValueError("Acoustic codes require one coordinate per family plus four acoustic coordinates")
    codes = torch.zeros(len(calls), latent_dim)
    for k, call in enumerate(calls):
        if (call.harmonic_count != config.harmonic_count
                or (call.harmonic_count > 1 and call.harmonic_rolloff != config.harmonic_rolloff)):
            raise ValueError("Call harmonic structure differs from the acoustic prior")
        codes[k, config.call_families.index(call.kind)] = 1
        values = (call.frequency, call.bandwidth, call.modulation_hz, call.pulse_hz)
        for j, (value, (lo, hi)) in enumerate(zip(values, acoustic_ranges(config))):
            if not lo <= value <= hi:
                raise ValueError("Call parameters lie outside the acoustic prior domain")
            codes[k, start+j] = 2 * (value-lo) / (hi-lo) - 1
    return codes


def template_codes(config: Config, frequency_step: float = 25.) -> Tensor:
    """A generic latent bank, constructed without test-call parameters."""
    lo, hi = acoustic_ranges(config)[0]
    frequencies = torch.arange(lo, hi + 1e-4, frequency_step)
    calls = []
    for f in frequencies.tolist():
        for b, m in ((180., 3.5), (180., 6.), (360., 3.5), (360., 6.)):
            if "tonal" in config.call_families:
                calls.append(CallSpec("tonal", config.call_duration, f, b * config.frequency_scale, m, 16.,
                                      config.harmonic_count, config.harmonic_rolloff))
        for b in (150., 300., 450.):
            if "sweep" in config.call_families:
                calls.append(CallSpec("sweep", config.call_duration, f, b * config.frequency_scale, 5., 16.,
                                      config.harmonic_count, config.harmonic_rolloff))
        for p in (12., 16., 20.):
            if "pulse" in config.call_families:
                calls.append(CallSpec("pulse", config.call_duration, f, 280. * config.frequency_scale, 5., p,
                                      config.harmonic_count, config.harmonic_rolloff))
        if "moo" in config.call_families:
            for b in (150., 300., 450.):
                calls.append(CallSpec("moo", config.call_duration, f, b * config.frequency_scale, 5., 16.,
                                      config.harmonic_count, config.harmonic_rolloff))
    return encode_calls(calls, config.latent_dim, config)


def _targets(calls: list[CallSpec], config: Config) -> tuple[Tensor, Tensor, Tensor]:
    fields = []
    for call in calls:
        power, times, frequencies = power_spectrogram(synthesize_call(call, config.sample_rate), config)
        fields.append(power)
    targets = torch.stack(fields)
    targets = targets / targets.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    return targets, times, frequencies


@torch.no_grad()
def evaluate_prior_calls(prior: FrozenCallPrior, calls: list[CallSpec], config: Config) -> tuple[Tensor, Tensor]:
    """Representation diagnostic with known codes; not part of mixture inference."""
    target, times, frequencies = _targets(calls, config)
    codes = encode_calls(calls, config.latent_dim, config)
    durations = torch.full((len(calls),), config.call_duration)
    predicted = evaluate_field(normalized_splauts(prior.generator, codes, durations, times, frequencies), times, frequencies)
    return target, predicted


@torch.no_grad()
def harmonic_reconstruction_metrics(prior: FrozenCallPrior, calls: list[CallSpec], config: Config) -> dict:
    """Known-code validation per order, so a strong fundamental cannot hide weak bands.

    Assign bins within half an instantaneous fundamental of each harmonic.
    This is a diagnostic partition of power, not waveform source separation.
    """
    target, prediction = evaluate_prior_calls(prior, calls, config)
    errors = [[] for _ in range(config.harmonic_count)]
    for k, call in enumerate(calls):
        t = prior.relative_times
        fundamental = instantaneous_frequency(call, t)
        ratio = prior.frequencies[None] / fundamental[:, None]
        for h in range(1, config.harmonic_count + 1):
            mask = (ratio >= h - 0.5) & (ratio < h + 0.5)
            error = ((prediction[k] - target[k])[mask].square().sum()
                     / target[k][mask].square().sum().clamp_min(1e-12))
            errors[h-1].append(float(error))
    return {"known_codes": True, "partition": "bins within half an instantaneous fundamental of each order",
            "mean_nmse_by_order": [sum(values) / len(values) for values in errors], "per_call_nmse_by_order": errors}


def load_frozen_prior(path: Path, config: Config | None = None) -> FrozenCallPrior:
    bundle = torch.load(path, weights_only=True)
    saved_config = Config(**bundle["config"])
    if config is not None:
        keys = ("sample_rate", "n_fft", "hop_length", "call_duration", "splauts", "latent_dim",
                "harmonic_count", "harmonic_rolloff", "frequency_scale", "call_families")
        if any(getattr(config, key) != getattr(saved_config, key) for key in keys):
            raise ValueError("Requested model/STFT configuration differs from the pretrained checkpoint")
    generator = CallPriorGenerator(saved_config)
    generator.load_state_dict(bundle["generator"])
    generator.eval().requires_grad_(False)
    fingerprint = decoder_fingerprint(generator)
    if fingerprint != bundle["fingerprint"]:
        raise ValueError("Decoder fingerprint does not match saved pretrained weights")
    return FrozenCallPrior(generator, bundle["bank_codes"], bundle["relative_times"], bundle["frequencies"], fingerprint)


def pretrain_call_prior(config: Config, per_family: int = 32, steps: int = 2000,
                        seed: int = 23, validation_seed: int = 29,
                        batch_size: int = 24, frequency_step: float = 25.,
                        log_every: int = 200) -> PriorTrainingResult:
    """Training calls supervise only the decoder; test scenes are never inputs.

    Known acoustic codes are supervised conditioning during pretraining. Their
    values and family are unknown and inferred from mixtures during localization.
    Validation is evaluated after training and does not select the checkpoint.
    """
    if steps < 0 or batch_size < 1 or frequency_step <= 0 or validation_seed == seed:
        raise ValueError("Use nonnegative steps, positive batch/bank spacing and distinct validation seed")
    calls = sample_training_calls(config, per_family, seed)
    codes = encode_calls(calls, config.latent_dim, config)
    targets, times, frequencies = _targets(calls, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = CallPriorGenerator(config)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.004)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=0.0001)
    rng = torch.Generator().manual_seed(seed + 1000)
    history = []
    durations = torch.full((len(calls),), config.call_duration)
    for step in range(steps + 1):
        indices = torch.randperm(len(calls), generator=rng)[:batch_size]
        splauts = normalized_splauts(generator, codes[indices], durations[indices], times, frequencies)
        prediction = evaluate_field(splauts, times, frequencies)
        mse = (prediction - targets[indices]).square().mean()
        floor = targets[indices].amax((1, 2), keepdim=True) * 1e-6 + 1e-8
        compressed = ((torch.sqrt(prediction + floor) - torch.sqrt(targets[indices] + floor))
                      .square().mean((1, 2)) / targets[indices].mean((1, 2))).mean()
        loss = mse + 0.2 * compressed
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite isolated-call pretraining loss")
        if step % max(log_every, 1) == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach()), "shape_nmse": float(mse.detach())})
            if log_every:
                print(f"  pretrain {step}/{steps}: shape NMSE={float(mse.detach()):.4f}", flush=True)
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.)
            optimizer.step()
            scheduler.step()
    generator.eval().requires_grad_(False)
    for parameter in generator.parameters():
        parameter.grad = None
    with torch.no_grad():
        def errors(evaluation_codes, evaluation_targets):
            duration = torch.full((len(evaluation_codes),), config.call_duration)
            prediction = evaluate_field(normalized_splauts(generator, evaluation_codes, duration, times, frequencies), times, frequencies)
            return (prediction - evaluation_targets).square().mean((1, 2)).tolist()
        training_errors = errors(codes, targets)
        validation_calls = sample_training_calls(config, max(8, per_family // 4), validation_seed)
        validation_codes = encode_calls(validation_calls, config.latent_dim, config)
        validation_targets, _, _ = _targets(validation_calls, config)
        validation_errors = errors(validation_codes, validation_targets)
    prior = FrozenCallPrior(generator, template_codes(config, frequency_step), times, frequencies,
                            decoder_fingerprint(generator))
    return PriorTrainingResult(prior, calls, codes, targets, history,
                                training_errors, validation_errors, validation_calls)
