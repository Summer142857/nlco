# pipeline/run.py
from __future__ import annotations

import argparse

from pipeline.config_loader import load_problem_scale_config
from problems import registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True, help="e.g., MIS")
    parser.add_argument("--scale", type=str, required=True, help="e.g., S / M / L")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional global config to merge first (problem config overrides it).",
    )
    args = parser.parse_args()

    cfg = load_problem_scale_config(
        problem=args.problem,
        scale=args.scale,
        global_config_path=args.config,
    )

    entry = registry.get_problem(args.problem)
    print(f"[PIPELINE] Running problem: {entry.name} (scale={args.scale})")

    raw_path = entry.generate(cfg)
    if entry.contextualize is not None:
        entry.contextualize(cfg)



if __name__ == "__main__":
    main()
