"""Mixture-only learning diagnostics, with ground truth used only after fitting."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

from .plots import _save
from ..model.generator import evaluate_field


@torch.no_grad()
def plot_components(result, frequencies, path: Path) -> None:
    fields = (evaluate_field(result.splauts, result.proposals.relative_times, frequencies)
              * result.state.amplitudes[:, None, None])
    count = len(fields)
    fig, axes = plt.subplots(count, 3, figsize=(12, 2.6 * count), squeeze=False)
    extent = [0, float(result.proposals.relative_times[-1]), 0, float(frequencies[-1])]
    for k in range(count):
        vmax = max(float(fields[k].max()), float(result.proposals.fields[k].max()), 1e-12)
        for ax, field, title in zip(axes[k, :2], (result.proposals.fields[k], fields[k]),
                                    ("Mixture-derived initialization", "Inferred emitted field")):
            db = 10 * np.log10(np.maximum(field.numpy() / vmax, 1e-5))
            ax.imshow(db.T, origin="lower", aspect="auto", extent=extent, vmin=-50, vmax=0)
            ax.set(title=f"Component {k+1}: {title}", xlabel="Relative time (s)", ylabel="Frequency (Hz)")
        splauts = result.splauts
        weights = splauts.weight[k]
        for n in range(len(weights)):
            axes[k, 2].add_patch(Ellipse((float(splauts.time[k, n]), float(splauts.frequency[k, n])),
                                         2 * float(splauts.time_width[k, n]), 2 * float(splauts.frequency_width[k, n]),
                                         facecolor=f"C{k}", edgecolor="black", linewidth=0.3,
                                         alpha=max(0.05, float(weights[n] / weights.max().clamp_min(1e-12)) * 0.7)))
        axes[k, 2].set(xlim=extent[:2], ylim=extent[2:], title=f"Component {k+1}: splauts (1σ)",
                       xlabel="Relative time (s)", ylabel="Frequency (Hz)")
    fig.suptitle("Source components inferred only from station mixtures", y=1.01)
    fig.tight_layout()
    _save(fig, path)


def plot_evaluation(scene, result, assignment: list[int], errors, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(*scene.stations.numpy().T, marker="^", c="black", s=60, label="Stations")
    for truth_index, component in enumerate(assignment):
        color = f"C{truth_index}"
        track = result.trajectories[:, component].numpy()
        ax.plot(*track.T, color=color, alpha=0.7)
        ax.scatter(*scene.events[truth_index].position.numpy(), marker="*", color=color, s=140)
        ax.scatter(*track[-1], marker="x", color=color, s=70,
                   label=f"True {truth_index+1} ↔ component {component+1}")
        axes[1].plot(errors[:, truth_index], color=color, label=f"True {truth_index+1}")
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal", title="True (*) and inferred (×) positions")
    ax.legend(fontsize=7)
    axes[1].set(xlabel="Mixture optimization step (last point: selected fit)", ylabel="Position error (m)",
                title="Errors using final position assignment")
    axes[1].legend(fontsize=7)
    fig.suptitle("Post-fit evaluation: test source labels were withheld during inversion")
    fig.tight_layout()
    _save(fig, path)


def plot_fit_losses(result, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    if result.warmup_losses:
        axes[0].semilogy(result.warmup_losses)
        axes[0].set(title="Initialization from mixture back-projections", xlabel="Warmup step", ylabel="Normalized field MSE")
    else:
        axes[0].text(0.5, 0.5, "Pretrained decoder frozen\nOnly event parameters fitted", ha="center", va="center")
        axes[0].set_axis_off()
    axes[1].semilogy(result.objectives, label="Objective (including extent prior)")
    axes[1].semilogy(result.power_losses, label="Station-balanced power NMSE")
    axes[1].semilogy(result.compressed_losses, label="Square-root-power loss")
    axes[1].axvline(result.best_step, color="grey", ls="--", label="Selected fit")
    axes[1].set(title="Mixture-only reconstruction", xlabel="Optimization step", ylabel="Loss")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def plot_sweep(rows: list[dict], path: Path) -> None:
    event_counts, station_counts = (2, 3, 4, 5), (4, 6, 8)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, key, title in zip(axes, ("mean_position_error_m", "power_nmse", "fraction_within_10m", "mean_source_power_nmse"),
                              ("Mean matched position error (m)", "Station-balanced power NMSE", "Fraction matched within 10 m", "Post-fit source power NMSE")):
        values = np.full((4, 3), np.nan)
        for row in rows:
            values[event_counts.index(row["events"]), station_counts.index(row["stations"])] = row.get(key, np.nan)
        im = ax.imshow(values, aspect="auto", cmap="viridis_r" if key != "fraction_within_10m" else "viridis")
        for iy, ix in np.ndindex(values.shape):
            if np.isfinite(values[iy, ix]):
                ax.text(ix, iy, f"{values[iy, ix]:.2f}", ha="center", va="center", color="white",
                        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none"})
        ax.set(xticks=range(3), xticklabels=station_counts, yticks=range(4), yticklabels=event_counts,
               xlabel="Stations", ylabel="Calls (count supplied)", title=title)
        fig.colorbar(im, ax=ax)
    frozen = bool(rows) and rows[0].get("variant") == "frozen_pretrained"
    title = ("Frozen pretrained decoder: held-out mixtures; test labels used only for evaluation" if frozen
             else "Mixture-only fitting: one fixed scene seed, labels used only for evaluation")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)


def plot_source_evaluation(truth, estimated, times, frequencies, path: Path) -> None:
    count = len(truth)
    fig, axes = plt.subplots(count, 2, figsize=(10, 2.5 * count), squeeze=False)
    extent = [float(times[0]), float(times[-1]), float(frequencies[0]), float(frequencies[-1])]
    for k in range(count):
        reference = max(float(truth[k].max()), float(estimated[k].max()), 1e-12)
        for ax, field, label in zip(axes[k], (truth[k], estimated[k]), ("Withheld source", "Matched inferred source")):
            db = 10 * np.log10(np.maximum(field.numpy() / reference, 1e-5))
            ax.imshow(db.T, origin="lower", aspect="auto", extent=extent, vmin=-50, vmax=0)
            ax.set(title=f"True event {k+1}: {label}", xlabel="Absolute emission time (s)", ylabel="Frequency (Hz)")
    fig.suptitle("Post-fit source comparison: held-out source fields were never inversion inputs", y=1.01)
    fig.tight_layout()
    _save(fig, path)
