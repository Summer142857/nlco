# pipeline/run_all.py
from __future__ import annotations

import argparse

from pipeline.config_loader import load_problem_scale_config
from problems import registry


def run_one(problem: str, scale: str, global_config: str | None = None):
    cfg = load_problem_scale_config(problem=problem, scale=scale, global_config_path=global_config)
    entry = registry.get_problem(problem)

    print(f"[PIPELINE] Running problem: {problem} (scale={scale})")
    raw_path = entry.generate(cfg)
    if entry.contextualize is not None:
        entry.contextualize(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="S", help="Run one scale for all problems, default S")
    parser.add_argument("--scales", type=str, default=None, help="Comma-separated scales, e.g. S,M,L")
    parser.add_argument("--only", type=str, default=None, help="Only run one problem, e.g. MIS")
    parser.add_argument("--config", type=str, default=None, help="Optional global config to merge first")
    args = parser.parse_args()

    if args.scales:
        scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    else:
        scales = [args.scale]

    if args.only:
        problems = [args.only]
    else:
        problems = list(registry.list_problem_names())

    for p in problems:
        for s in scales:
            run_one(problem=p, scale=s, global_config=args.config)


if __name__ == "__main__":
    main()
