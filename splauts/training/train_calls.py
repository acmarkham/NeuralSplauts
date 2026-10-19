"""Fit isolated procedural calls with a shared decoder and per-example embeddings."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import Config
from ..model.generator import SplautGenerator, evaluate_field
from ..simulator.calls import CallSpec, power_spectrogram, synthesize_call


@dataclass
class CallTrainingResult:
    generator: SplautGenerator
    latents: Tensor
    durations: Tensor
    targets: Tensor
    times: Tensor
    frequencies: Tensor
    losses: list[float]  # Raw power-domain MSE, including initial and final states.


def train_calls(calls: list[CallSpec], config: Config, steps: int = 600,
                learning_rate: float = 0.005) -> CallTrainingResult:
    if not calls or steps < 0 or learning_rate <= 0:
        raise ValueError("Supply calls, nonnegative steps and a positive learning rate")
    if any(abs(c.duration - config.call_duration) > 1e-6 for c in calls):
        raise ValueError("Batch training currently uses config.call_duration for every call")
    # Keep callers' RNG state intact while making initialization reproducible.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        generator = SplautGenerator(config)
        latents = nn.Parameter(torch.randn(len(calls), config.latent_dim) * 0.1)
    targets = []
    for call in calls:
        power, times, frequencies = power_spectrogram(synthesize_call(call, config.sample_rate), config)
        targets.append(power)
    target = torch.stack(targets)
    generator.initialize_from_power(target, times, frequencies, config.call_duration, config.seed)
    durations = torch.tensor([c.duration for c in calls])
    optimizer = torch.optim.Adam([*generator.parameters(), latents], lr=learning_rate)
    scale = generator.intensity_scale.detach()
    losses = []
    for step in range(steps + 1):
        prediction = evaluate_field(generator(latents, durations), times, frequencies)
        mse = (prediction - target).square().mean()
        losses.append(float(mse.detach()))
        if step < steps:
            optimizer.zero_grad()
            (mse / scale.square()).backward()
            optimizer.step()
    generator.eval()
    return CallTrainingResult(generator, latents.detach(), durations, target, times, frequencies, losses)
