"""Pretraining quality and comparison with the mixture-trained baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import torch

from ..training.pretrain import encode_calls
from ..training.mixtures import normalized_splauts

from .plots import _save


def plot_prior_quality(training, target, predicted, path: Path) -> None:
    config = training.prior.generator.config
    families = config.call_families
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].semilogy([r["step"] for r in training.training_history],
                     [r["shape_nmse"] for r in training.training_history])
    axes[0].set(title="Supervised single-call pretraining", xlabel="Minibatch step", ylabel="Shape NMSE")
    values = np.asarray(training.validation_nmse).reshape(len(families), -1)
    axes[1].bar(families, values.mean(1))
    axes[1].set(title="Independent isolated validation calls (known codes)", ylabel="Mean shape NMSE")
    fig.tight_layout()
    _save(fig, path / "pretraining.png")
    count = len(target) // len(families)
    fig, axes = plt.subplots(len(families), 3, figsize=(15, 2.7 * len(families)), squeeze=False)
    extent = [0, float(training.prior.relative_times[-1]), 0, float(training.prior.frequencies[-1])]
    config = training.prior.generator.config
    with torch.no_grad():
        codes = encode_calls(training.validation_calls, config.latent_dim, config)
        atoms = normalized_splauts(training.prior.generator, codes,
                                   torch.full((len(codes),), config.call_duration),
                                   training.prior.relative_times, training.prior.frequencies)
    for k, kind in enumerate(families):
        index = k * count
        maximum = max(float(target[index].max()), float(predicted[index].max()), 1e-8)
        for ax, field, title in zip(axes[k, :2], (target[index], predicted[index]), ("True isolated validation call", "Frozen decoder")):
            db = 10 * np.log10(np.maximum(field.numpy() / maximum, 1e-5))
            ax.imshow(db.T, origin="lower", aspect="auto", extent=extent, vmin=-50, vmax=0)
            ax.set(title=f"{kind}: {title}", xlabel="Relative time (s)", ylabel="Frequency (Hz)")
        ax = axes[k, 2]
        maximum_weight = float(atoms.weight[index].max())
        for n in range(config.splauts):
            strength = float(atoms.weight[index, n]) / maximum_weight
            if strength < 0.005:
                continue
            order = n % config.harmonic_count
            ax.add_patch(Ellipse((float(atoms.time[index, n]), float(atoms.frequency[index, n])),
                                 4 * float(atoms.time_width[index, n]), 4 * float(atoms.frequency_width[index, n]),
                                 facecolor=f"C{order % 10}", edgecolor=f"C{order % 10}", alpha=max(0.1, strength * 0.7)))
        ax.set(xlim=extent[:2], ylim=extent[2:], title="Splauts in one atomic event (2σ)",
               xlabel="Relative time (s)", ylabel="Frequency (Hz)")
    fig.suptitle(f"{config.harmonic_count} harmonic(s) per call; one code, position and onset per event. Known-code validation.")
    fig.tight_layout()
    _save(fig, path / "validation_calls.png")


def plot_prior_comparison(rows, baseline, path: Path) -> None:
    reference = {(r["events"], r["stations"]): r for r in baseline}
    matched = [r for r in rows if (r["events"], r["stations"]) in reference]
    if not matched:
        return
    labels = [f"{r['events']} calls / {r['stations']} stations" for r in matched]
    y = np.arange(len(matched))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, key, title in zip(axes, ("mean_position_error_m", "mean_source_power_nmse"),
                              ("Mean matched position error (m)", "Mean source-field NMSE")):
        old = [reference[(r["events"], r["stations"])][key] for r in matched]
        new = [r[key] for r in matched]
        ax.barh(y-0.18, old, height=0.36, label="Mixture-trained decoder")
        ax.barh(y+0.18, new, height=0.36, label="Frozen supervised prior")
        ax.set(yticks=y, yticklabels=labels, xlabel=title)
        ax.legend(fontsize=8)
    fig.suptitle("Same held-out scenes; decoder parameterization and candidate initialization also differ")
    fig.tight_layout()
    _save(fig, path)
