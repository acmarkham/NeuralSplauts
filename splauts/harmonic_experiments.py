"""Post-fit paired comparison: python -m splauts.harmonic_experiments."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .mixture_experiments import _write_json
from .simulator.scene import random_scene
from .visualization.plots import _save


def compare_harmonic_sweeps(harmonic: Path, control: Path, output: Path) -> dict:
    manifests = [json.loads((path / "results.json").read_text()) for path in (control, harmonic)]
    metadata = [json.loads((path / "pretraining.json").read_text()) for path in (control, harmonic)]
    if metadata[0]["config"]["harmonic_count"] != 1 or metadata[1]["config"]["harmonic_count"] <= 1:
        raise ValueError("Use a one-harmonic control and a multi-harmonic experiment")
    for manifest in manifests:
        if manifest["completed"] != manifest["requested"] or not manifest["cases"]:
            raise ValueError("Both sweeps must be complete before comparison")
    for key in ("training_seed", "validation_seed", "training_examples", "validation_examples", "pretrain_steps", "bank_frequency_step_hz"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f"Paired pretraining setting differs: {key}")
    indexed = [{(r["events"], r["stations"]): r for r in m["cases"]} for m in manifests]
    if indexed[0].keys() != indexed[1].keys():
        raise ValueError("Paired sweeps must have the same case matrix")
    cases = []
    metrics = ("position_errors_m", "mean_position_error_m", "fraction_within_10m", "power_nmse",
               "source_power_nmse", "mean_source_power_nmse", "decoder_unchanged", "test_calls_disjoint_from_training")
    for key in sorted(indexed[0]):
        a, b = [rows[key] for rows in indexed]
        for setting, value in a["config"].items():
            if setting != "harmonic_count" and b["config"][setting] != value:
                raise ValueError(f"Paired scene/model setting differs: {setting}")
        for setting in ("steps", "proposal_steps", "proposal_grid_size", "true_positions_m", "true_onsets_s"):
            if a[setting] != b[setting]:
                raise ValueError(f"Paired fitting/geometry setting differs: {setting}")
        cases.append({"events": key[0], "stations": key[1],
                      "control": {k: a[k] for k in metrics}, "harmonic": {k: b[k] for k in metrics}})
    summary = {}
    for label, meta, path in zip(("control", "harmonic"), metadata, (control, harmonic)):
        summary[label] = {"config": meta["config"], "mean_training_nmse": meta["mean_training_nmse"],
                          "mean_validation_nmse": meta["mean_validation_nmse"],
                          "localized_within_10m": sum(sum(e <= 10 for e in r[label]["position_errors_m"]) for r in cases),
                          "total_sources": sum(r["events"] for r in cases),
                          "all_decoders_unchanged": all(r[label]["decoder_unchanged"] for r in cases),
                          "all_test_calls_disjoint_from_training": all(r[label]["test_calls_disjoint_from_training"] for r in cases),
                          **{k: float(np.mean([r[label][k] for r in cases]))
                             for k in ("power_nmse", "mean_source_power_nmse")}}
        bands = meta.get("validation_harmonic_metrics")
        if bands is None and (path / "harmonic_validation.json").exists():
            bands = json.loads((path / "harmonic_validation.json").read_text())
        if bands is not None:
            summary[label]["validation_nmse_by_harmonic_order"] = bands["mean_nmse_by_order"]
    # Post-fit input audit; clean calls never enter either inversion.
    scene_count = max(r["events"] for r in cases)
    scenes = [random_scene(Config(**m["config"]), scene_count, 4) for m in metadata]
    energy_ratios = [float(b.square().mean() / a.square().mean())
                    for a, b in zip(scenes[0].source_waveforms, scenes[1].source_waveforms)]
    report = {"protocol": {"scene_seed": metadata[0]["config"]["seed"],
                           "pretrain_steps": metadata[0]["pretrain_steps"],
                           "training_examples": metadata[0]["training_examples"],
                           "validation_examples": metadata[0]["validation_examples"],
                           "bank_frequency_step_hz": metadata[0]["bank_frequency_step_hz"],
                           "mixture_steps": manifests[0]["cases"][0]["steps"],
                           "proposal_steps": manifests[0]["cases"][0]["proposal_steps"],
                           "proposal_grid_size": manifests[0]["cases"][0]["proposal_grid_size"],
                           "training_seed": metadata[0]["training_seed"], "validation_seed": metadata[0]["validation_seed"],
                           "energy_control": "unit sum of squared oscillator amplitudes; approximately equal waveform RMS",
                           "harmonic_to_control_source_energy_ratios": energy_ratios,
                           "held_equal": "fundamentals, excursions, geometry, onsets, source amplitudes, seeds, total splauts, budgets and bank spacing",
                           "limitation": "one scene seed; learned shapes and greedy inversion can fail; count and duration supplied"},
              "summary": summary, "cases": cases}
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "comparison.json", report)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    y = np.arange(len(cases))
    labels = [f"{r['events']} calls / {r['stations']} stations" for r in cases]
    for ax, metric, title in zip(axes, ("fraction_within_10m", "power_nmse", "mean_source_power_nmse"),
                                  ("Fraction localized within 10 m", "Mixture power NMSE", "Source-field NMSE")):
        for offset, label, name in ((-0.18, "control", "One harmonic"),
                                    (0.18, "harmonic", f"{metadata[1]['config']['harmonic_count']} harmonics")):
            ax.barh(y + offset, [r[label][metric] for r in cases], height=0.36, label=name)
        ax.set(yticks=y, yticklabels=labels, xlabel=title)
        ax.legend(fontsize=8)
    fig.suptitle(f"Paired seed-{metadata[0]['config']['seed']} scenes: same fundamentals, energy, geometry and total splaut capacity")
    fig.tight_layout()
    _save(fig, output / "comparison.png")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonic", type=Path, default=Path("outputs/harmonic4"))
    parser.add_argument("--control", type=Path, default=Path("outputs/harmonic_control"))
    parser.add_argument("--output", type=Path, default=Path("outputs/harmonic_comparison"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    result = compare_harmonic_sweeps(args.harmonic, args.control, args.output)
    print(json.dumps(result["summary"], indent=2))
