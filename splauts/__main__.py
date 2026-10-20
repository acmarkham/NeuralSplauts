"""Run with python -m splauts demo, compare, mixtures, or pretrained."""

import argparse
from pathlib import Path

import torch

from .config import Config
from .experiments import run_comparison, run_demo
from .mixture_experiments import run_mixture_sweep
from .pretrained_experiments import run_pretrained_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="experiment", required=True)
    demo = commands.add_parser("demo", help="Simulate, learn calls and localize 1–3 associated events")
    comparison = commands.add_parser("compare", help="Compare fixed geometry/noise/call/station-count cases")
    mixtures = commands.add_parser("mixtures", help="Learn splauts from mixtures only; default is the full 4×3 sweep")
    pretrained = commands.add_parser("pretrained", help="Supervise isolated calls, freeze the decoder, and invert held-out mixtures")
    for command in (demo, comparison, mixtures, pretrained):
        command.add_argument("--output", type=Path)
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--splauts", type=int, default=32)
        command.add_argument("--latent-dim", type=int, default=16)
        command.add_argument("--threads", type=int, default=4 if command in (mixtures, pretrained) else 1)
    for command in (demo, comparison):
        command.add_argument("--train-steps", type=int, default=600)
        command.add_argument("--localize-steps", type=int, default=250)
        command.add_argument("--grid-size", type=int, default=25)
    for command in (mixtures, pretrained):
        command.add_argument("--events", type=int, nargs="+", choices=(2, 3, 4, 5), default=[2, 3, 4, 5])
        command.add_argument("--stations", type=int, nargs="+", choices=(4, 6, 8), default=[4, 6, 8])
        command.add_argument("--steps", type=int, default=400)
        command.add_argument("--proposal-grid-size", type=int, default=25 if command is pretrained else 13)
        command.add_argument("--noise", type=float, default=0.0)
        command.add_argument("--harmonics", type=int, default=1, help="Harmonic orders in each atomic call")
        command.add_argument("--harmonic-rolloff", type=float, default=0.7, help="Amplitude of order h is h**(-rolloff)")
        command.add_argument("--frequency-scale", type=float, help="Scale fundamental and bandwidth; default min(1, 1.2/harmonics)")
        command.add_argument("--moo", action="store_true", help="Include low-high-low moo calls; scenes cycle moo/tonal/sweep/pulse")
        command.add_argument("--absorption-db-per-km", type=float, default=0., help="Spectral absorption at 1 kHz in dB/km")
        command.add_argument("--absorption-power", type=float, default=2., help="Frequency exponent of absorption")
        command.add_argument("--absorption-distance-bins", type=int, default=8, help="Spectral-template distance bins for proposal search")
        command.add_argument("--call-duration", type=float, default=0.4)
        command.add_argument("--scene-duration", type=float, default=2.4)
    mixtures.add_argument("--warmup-steps", type=int, default=250)
    mixtures.add_argument("--geometry-steps", type=int, default=75)
    mixtures.add_argument("--fit-seed", type=int, default=101)
    pretrained.add_argument("--pretrain-steps", type=int, default=2000)
    pretrained.add_argument("--examples-per-family", type=int, default=32)
    pretrained.add_argument("--pretrain-seed", type=int, default=23)
    pretrained.add_argument("--validation-seed", type=int, default=29)
    pretrained.add_argument("--bank-step", type=float, help="Generic fundamental spacing in Hz; default 25 times frequency-scale")
    pretrained.add_argument("--proposal-steps", type=int, default=80)
    pretrained.add_argument("--prior", type=Path, help="Reuse a saved frozen prior instead of pretraining again")
    pretrained.add_argument("--pretrain-only", action="store_true")
    pretrained.add_argument("--resume", action="store_true", help="Retain compatible completed fits and run remaining cases")
    demo.add_argument("--events", type=int, choices=(1, 2, 3), default=1)
    demo.add_argument("--stations", type=int, choices=(4, 6, 8), default=4)
    demo.add_argument("--layout", choices=("corners", "irregular", "clustered"), default="corners")
    demo.add_argument("--call", choices=("tonal", "sweep", "pulse"), default=None)
    demo.add_argument("--noise", type=float, default=0.0, help="Absolute station waveform noise std")
    demo.add_argument("--refine", action="store_true", help="Also progressively optimize onset, amplitude, embedding")
    demo.add_argument("--coarse-grid-size", type=int, default=13,
                      help="Data-only multi-event initialization grid; 0 disables it")
    args = parser.parse_args()
    if args.experiment == "pretrained":
        if (args.threads < 1 or args.proposal_grid_size < 2 or args.examples_per_family < 1
                or (args.bank_step is not None and args.bank_step <= 0) or args.latent_dim < 7
                or min(args.pretrain_steps, args.steps, args.proposal_steps) < 0):
            parser.error("Use positive threads/examples/bank spacing, grid >=2, latent_dim >=7 and nonnegative steps")
        torch.set_flush_denormal(True)
        torch.set_num_threads(args.threads)
        config = mixture_config(args, parser)
        bank_step = args.bank_step if args.bank_step is not None else 25. * config.frequency_scale
        output = args.output or variant_output("pretrained", args)
        run_pretrained_sweep(config, output, args.events, args.stations, args.pretrain_steps,
                              args.examples_per_family, args.steps, args.proposal_steps, args.proposal_grid_size,
                              args.pretrain_seed, args.validation_seed, bank_step, args.prior, args.pretrain_only, args.resume)
        print(f"Saved results to {output.resolve()}")
        return
    if args.experiment == "mixtures":
        if (args.threads < 1 or args.proposal_grid_size < 2
                or min(args.warmup_steps, args.steps, args.geometry_steps) < 0):
            parser.error("Use positive threads, proposal grid >=2 and nonnegative iteration counts")
        # Gaussian tails produce subnormal float32 values that severely slow CPU
        # arithmetic. Their powers are far below the numerical loss floors.
        torch.set_flush_denormal(True)
        torch.set_num_threads(args.threads)
        config = mixture_config(args, parser)
        output = args.output or variant_output("mixtures", args)
        run_mixture_sweep(config, output, args.events, args.stations, args.warmup_steps,
                          args.steps, args.proposal_grid_size, args.fit_seed, args.geometry_steps)
        print(f"Saved results to {output.resolve()}")
        return
    if min(args.threads, args.grid_size) < 1 or args.grid_size < 2 or min(args.train_steps, args.localize_steps) < 0:
        parser.error("Use positive threads, grid size >=2 and nonnegative iteration counts")
    if getattr(args, "coarse_grid_size", 0) < 0 or getattr(args, "coarse_grid_size", 0) == 1:
        parser.error("coarse-grid-size must be 0 or at least 2")
    torch.set_flush_denormal(True)
    torch.set_num_threads(args.threads)
    config = Config(noise_std=getattr(args, "noise", 0.0), seed=args.seed, splauts=args.splauts, latent_dim=args.latent_dim)
    output = args.output or Path("outputs") / args.experiment
    if args.experiment == "demo":
        run_demo(config, output, args.events, args.stations, args.layout, args.call,
                 args.train_steps, args.localize_steps, args.grid_size, args.refine, args.coarse_grid_size)
    else:
        run_comparison(config, output, args.train_steps, args.localize_steps, args.grid_size)
    print(f"Saved results to {output.resolve()}")


def variant_output(prefix: str, args) -> Path:
    suffix = "" if args.harmonics == 1 else f"_harmonic{args.harmonics}"
    suffix += "_moo" if args.moo else ""
    suffix += f"_absorption{args.absorption_db_per_km:g}" if args.absorption_db_per_km else ""
    return Path("outputs") / (prefix + suffix)


def mixture_config(args, parser) -> Config:
    if args.harmonics < 1:
        parser.error("harmonics must be a positive integer")
    scale = args.frequency_scale if args.frequency_scale is not None else min(1., 1.2 / args.harmonics)
    try:
        return Config(noise_std=args.noise, seed=args.seed, splauts=args.splauts, latent_dim=args.latent_dim,
                      harmonic_count=args.harmonics, harmonic_rolloff=args.harmonic_rolloff, frequency_scale=scale,
                      call_families=("moo", "tonal", "sweep", "pulse") if args.moo else ("tonal", "sweep", "pulse"),
                      absorption_db_per_km=args.absorption_db_per_km, absorption_power=args.absorption_power,
                      absorption_distance_bins=args.absorption_distance_bins,
                      call_duration=args.call_duration, scene_duration=args.scene_duration)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
