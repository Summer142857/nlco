"""
Unified CLI for generating combinatorial optimisation instances.

Usage examples::

    # List all registered problems
    python -m step1_instance_creation.generate_cli --list

    # Generate MIS Small (50 instances)
    python -m step1_instance_creation.generate_cli \\
        --problem MIS --sizes S \\
        --output_root ./generated_data \\
        --dataset_root ./step1_instance_creation

    # Generate multiple problems / sizes
    python -m step1_instance_creation.generate_cli \\
        --problem MIS,MVC,MAXCUT --sizes S,M,L \\
        --output_root ./generated_data \\
        --n_instances 50 --seed 42

    # Generate everything
    python -m step1_instance_creation.generate_cli \\
        --problem all --sizes all \\
        --output_root ./generated_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .registry import PROBLEM_REGISTRY, ALL_PROBLEMS, get_problem_entry


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m step1_instance_creation.generate_cli",
        description="Generate CO benchmark instances for registered problems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--problem",
        default="all",
        help=(
            "Comma-separated problem names (e.g. MIS,CVRP,JSP) or 'all'. "
            "Use --list to see available names."
        ),
    )
    p.add_argument(
        "--sizes",
        default="all",
        help="Comma-separated size tiers: S,M,L or 'all' (default: all).",
    )
    p.add_argument(
        "--output_root",
        default="./generated_data",
        help="Root directory for output JSON files (default: ./generated_data).",
    )
    p.add_argument(
        "--dataset_root",
        default="./step1_instance_creation",
        help=(
            "Root directory that contains the datasets/ sub-folder "
            "(default: ./step1_instance_creation)."
        ),
    )
    p.add_argument(
        "--n_instances",
        type=int,
        default=None,
        help="Override the default instance count per size tier.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to every generator (default: 42).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print all registered problem names and exit.",
    )

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.list:
        print("Registered problems:")
        for name in ALL_PROBLEMS:
            entry = PROBLEM_REGISTRY[name]
            sizes_info = ", ".join(
                f"{sz}:{sc.size_range}" for sz, sc in entry.sizes.items()
            )
            print(f"  {name:12s}  sizes: {sizes_info}")
        return

    # ── Resolve problem list ────────────────────────────────────────────────
    if args.problem.lower() == "all":
        problems = ALL_PROBLEMS
    else:
        problems = [p.strip() for p in args.problem.split(",") if p.strip()]
        unknown = [p for p in problems if p not in PROBLEM_REGISTRY]
        if unknown:
            print(f"[Error] Unknown problem(s): {unknown}", file=sys.stderr)
            print(f"  Available: {ALL_PROBLEMS}", file=sys.stderr)
            sys.exit(1)

    # ── Resolve size list ───────────────────────────────────────────────────
    if args.sizes.lower() == "all":
        sizes = ["S", "M", "L"]
    else:
        sizes = [s.strip().upper() for s in args.sizes.split(",") if s.strip()]
        invalid = [s for s in sizes if s not in ("S", "M", "L")]
        if invalid:
            print(f"[Error] Invalid size(s): {invalid} — must be S, M, or L.", file=sys.stderr)
            sys.exit(1)

    output_root = Path(args.output_root)
    dataset_root = str(args.dataset_root)

    total = len(problems) * len(sizes)
    done = 0

    for prob_name in problems:
        entry = get_problem_entry(prob_name)
        for size in sizes:
            if size not in entry.sizes:
                print(f"[Warn] {prob_name} has no size tier '{size}', skipping.")
                continue

            out_path = output_root / prob_name / f"{prob_name}_{size}.json"
            print(
                f"[{done + 1}/{total}] Generating {prob_name} size={size} "
                f"-> {out_path}"
            )

            try:
                entry.generate(
                    size=size,
                    out_path=str(out_path),
                    n_instances=args.n_instances,
                    dataset_root=dataset_root,
                    seed=args.seed,
                )
            except Exception as exc:
                print(f"[Error] {prob_name}/{size} failed: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc()

            done += 1

    print(f"\nDone. Generated {done}/{total} problem/size combinations.")


if __name__ == "__main__":
    main()
