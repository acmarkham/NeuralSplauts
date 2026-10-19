"""Inspectable maps, individual splauts, station residuals and loss landscapes."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

from ..model.generator import evaluate_field


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_map(scene, result, area_size: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    stations = scene.stations.numpy()
    ax.scatter(*stations.T, marker="^", c="black", s=90, label="Stations")
    for m, station in enumerate(stations):
        ax.annotate(f"S{m+1}", station, xytext=(4, 4), textcoords="offset points")
    for k, event in enumerate(scene.events):
        color = f"C{k}"
        trajectory = result.trajectories[:, k].numpy()
        ax.plot(*trajectory.T, color=color, alpha=0.7)
        ax.scatter(*trajectory[0], marker="o", facecolors="none", edgecolors=color)
        ax.scatter(*event.position.numpy(), marker="*", c=color, s=160, label=f"Event {k+1}: true")
        ax.scatter(*trajectory[-1], marker="x", c=color, s=80, label=f"Event {k+1}: estimate")
    ax.set(xlim=(-15, area_size+15), ylim=(-15, area_size+15), xlabel="x (m)", ylabel="y (m)",
           title="Localization trajectories (circles mark initialization)", aspect="equal")
    ax.legend(fontsize=8)
    _save(fig, path)


@torch.no_grad()
def plot_calls(training, path: Path) -> None:
    splauts = training.generator(training.latents, training.durations)
    prediction = evaluate_field(splauts, training.times, training.frequencies)
    count = len(training.targets)
    fig, axes = plt.subplots(count, 3, figsize=(12, 3 * count), squeeze=False)
    extent = [float(training.times[0]), float(training.times[-1]),
              float(training.frequencies[0]), float(training.frequencies[-1])]
    for k in range(count):
        vmax = float(training.targets[k].max())
        for ax, field, title in zip(axes[k, :2], (training.targets[k], prediction[k]),
                                    ("Procedural STFT power", "Neural splaut field")):
            ax.imshow(field.T.numpy(), origin="lower", aspect="auto", extent=extent,
                      vmin=0, vmax=vmax)
            ax.set(title=f"Event {k+1}: {title}", xlabel="Relative time (s)", ylabel="Frequency (Hz)")
        ax = axes[k, 2]
        weights = splauts.weight[k]
        for n in range(len(weights)):
            # Ellipse radii are one standard deviation; opacity shows relative weight.
            ellipse = Ellipse((float(splauts.time[k, n]), float(splauts.frequency[k, n])),
                              2 * float(splauts.time_width[k, n]),
                              2 * float(splauts.frequency_width[k, n]),
                              alpha=max(0.08, float(weights[n] / weights.max()) * 0.7),
                              facecolor=f"C{k}", edgecolor="black", linewidth=0.4)
            ax.add_patch(ellipse)
        ax.set(xlim=extent[:2], ylim=extent[2:], title="Individual splauts (1σ contours)",
               xlabel="Relative time (s)", ylabel="Frequency (Hz)")
    fig.tight_layout()
    _save(fig, path)


def plot_stations(scene, prediction, path: Path) -> None:
    count = len(scene.stations)
    fig, axes = plt.subplots(count, 3, figsize=(12, 2.3 * count), squeeze=False)
    reference = float(scene.power.max())
    residual_limit = max(float((prediction - scene.power).abs().max()), 1e-20)
    extent = [float(scene.times[0]), float(scene.times[-1]), 0, float(scene.frequencies[-1])]
    for m in range(count):
        observed, predicted = scene.power[m].numpy(), prediction[m].numpy()
        # Shared log reference across all stations preserves amplitude differences.
        for ax, data, title in zip(axes[m, :2], (observed, predicted), ("Observed", "Rendered")):
            db = 10 * np.log10(np.maximum(data / max(reference, 1e-20), 1e-6))
            artist = ax.imshow(db.T, origin="lower", aspect="auto", extent=extent, vmin=-60, vmax=0)
            ax.set(title=f"S{m+1}: {title}", xlabel="Time (s)", ylabel="Frequency (Hz)")
            fig.colorbar(artist, ax=ax, label="dB relative to scene peak")
        residual = predicted - observed
        artist = axes[m, 2].imshow(residual.T, origin="lower", aspect="auto", extent=extent,
                                   cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit)
        axes[m, 2].set(xlabel="Time (s)", ylabel="Frequency (Hz)")
        axes[m, 2].set_title(f"S{m+1}: Rendered − observed", pad=16)
        fig.colorbar(artist, ax=axes[m, 2], label="Power residual")
    fig.tight_layout()
    _save(fig, path)


def plot_losses(training, runs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].semilogy(training.losses)
    axes[0].set(title="Call representation", xlabel="Iteration", ylabel="Power MSE")
    for name, result in runs.items():
        axes[1].semilogy(result.reconstruction_losses, label=name)
        if result.location_errors is not None:
            axes[2].plot(result.location_errors.mean(-1).numpy(), label=name)
    axes[1].set(title="Station reconstruction", xlabel="Iteration", ylabel="Normalized power MSE")
    axes[2].set(title="Localization", xlabel="Iteration", ylabel="Mean position error (m)")
    axes[1].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def plot_landscape(axis, losses, scene, result, path: Path, event_index: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    log_losses = np.log10(losses.numpy().clip(1e-8))
    # A source sitting on a station has extreme gain. Keep those few cells from
    # flattening every interior valley into the same colour; retain raw losses.
    upper = max(float(np.quantile(log_losses, 0.95)), float(log_losses.min()) + 1e-6)
    artist = ax.pcolormesh(axis.numpy(), axis.numpy(), log_losses, shading="nearest",
                           vmin=float(log_losses.min()), vmax=upper)
    fig.colorbar(artist, ax=ax, extend="max", label="log10(normalized MSE); upper 5% colour-clipped")
    ax.scatter(*scene.stations.numpy().T, marker="^", c="white", edgecolors="black", label="Stations")
    ax.scatter(*scene.events[event_index].position.numpy(), marker="*", c="red", s=140, label="True")
    trajectory = result.trajectories[:, event_index].numpy()
    ax.plot(*trajectory.T, c="orange", linewidth=1.5, label="Optimization")
    ax.scatter(*trajectory[-1], marker="x", c="orange", s=70, label="Estimated")
    ax.set(title=f"Event {event_index+1}: conditional spatial loss (others fixed)", xlabel="x (m)", ylabel="y (m)",
           xlim=(float(axis[0]), float(axis[-1])), ylim=(float(axis[0]), float(axis[-1])), aspect="equal")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_comparison(rows: list[dict], path: Path) -> None:
    labels = [r["case"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(labels, [r["error_m"] for r in rows])
    axes[1].barh(labels, [r["reconstruction_mse"] for r in rows])
    axes[0].set(xlabel="Position error (m)", title="Unsupervised-position localization")
    axes[1].set(xlabel="Normalized power MSE", title="Reconstruction")
    fig.tight_layout()
    _save(fig, path)
