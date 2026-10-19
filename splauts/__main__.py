"""Run with python -m splauts demo or python -m splauts compare."""

import argparse
from pathlib import Path

import torch

from .config import Config
from .experiments import run_comparison, run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="experiment", required=True)
    demo = commands.add_parser("demo", help="Simulate, learn calls and localize 1–3 associated events")
    comparison = commands.add_parser("compare", help="Compare fixed geometry/noise/call/station-count cases")
    for command in (demo, comparison):
        command.add_argument("--output", type=Path)
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--splauts", type=int, default=32)
        command.add_argument("--latent-dim", type=int, default=16)
        command.add_argument("--train-steps", type=int, default=600)
        command.add_argument("--localize-steps", type=int, default=250)
        command.add_argument("--grid-size", type=int, default=25)
        command.add_argument("--threads", type=int, default=1, help="Small CPU tensors usually work best with one thread")
    demo.add_argument("--events", type=int, choices=(1, 2, 3), default=1)
    demo.add_argument("--stations", type=int, choices=(4, 6, 8), default=4)
    demo.add_argument("--layout", choices=("corners", "irregular", "clustered"), default="corners")
    demo.add_argument("--call", choices=("tonal", "sweep", "pulse"), default=None)
    demo.add_argument("--noise", type=float, default=0.0, help="Absolute station waveform noise std")
    demo.add_argument("--refine", action="store_true", help="Also progressively optimize onset, amplitude, embedding")
    demo.add_argument("--coarse-grid-size", type=int, default=13,
                      help="Data-only multi-event initialization grid; 0 disables it")
    args = parser.parse_args()
    if min(args.threads, args.grid_size) < 1 or args.grid_size < 2 or min(args.train_steps, args.localize_steps) < 0:
        parser.error("Use positive threads, grid size >=2 and nonnegative iteration counts")
    if getattr(args, "coarse_grid_size", 0) < 0 or getattr(args, "coarse_grid_size", 0) == 1:
        parser.error("coarse-grid-size must be 0 or at least 2")
    torch.set_num_threads(args.threads)
    config = Config(noise_std=getattr(args, "noise", 0.0), seed=args.seed, splauts=args.splauts, latent_dim=args.latent_dim)
    output = args.output or Path("outputs") / args.experiment
    if args.experiment == "demo":
        run_demo(config, output, args.events, args.stations, args.layout, args.call,
                 args.train_steps, args.localize_steps, args.grid_size, args.refine, args.coarse_grid_size)
    else:
        run_comparison(config, output, args.train_steps, args.localize_steps, args.grid_size)
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
