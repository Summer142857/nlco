"""
Estimate empirical L_n on your dataset distribution.

- Reads your TSPLIB-like text (one Python-list per line):
  ["instance_name", best_obj_or_dummy, x1, y1, x2, y2, ...]
- For each n (or n-range buckets), samples random subsets of nodes from random instances,
  solves TSP via elkai (LKH wrapper), and records tour length.
- Reports per-n (or per-bucket) average tour length \hat{L}_n on YOUR distribution.

Dependencies:
  pip install elkai numpy

Usage examples:
  python estimate_Ln.py --dataset datasets/TSPlib_70instances.txt --n 5-30 --samples_per_n 50 --normalize --out Ln_stats.json
  python estimate_Ln.py --dataset datasets/TSP/ --n 5-10,11-20,21-30 --samples_per_n 100 --normalize --out Ln_buckets.json
"""

import ast
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence

import numpy as np
import elkai


# ---------------------------
# IO & parsing
# ---------------------------

def get_random_dataset_file(dataset_path: str) -> str:
    p = Path(dataset_path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        candidates: List[Path] = []
        for pat in ("*.tsp", "*.txt", "*.dat"):
            candidates += list(p.glob(pat))
        if not candidates:
            raise ValueError(f"No dataset files found under {dataset_path}")
        return str(random.choice(candidates))
    raise ValueError(f"Dataset path not found: {dataset_path}")


class TSPDatasetParser:
    """Reads your TSPLIB-like line format."""
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.instances = self._load_instances()

    def _load_instances(self) -> List[Dict[str, Any]]:
        instances: List[Dict[str, Any]] = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = ast.literal_eval(line)  # safe parse Python-literal list
                    name = data[0]
                    # data[1] is often optimal value (not used here)
                    coords_flat = [float(x) for x in data[2:]]
                    coords = np.array(list(zip(coords_flat[0::2], coords_flat[1::2])), dtype=float)
                    instances.append({"name": name, "coordinates": coords})
                except Exception as e:
                    print(f"[WARN] Failed to parse line head='{line[:60]}...' ({e})")
        if not instances:
            raise ValueError(f"No instances parsed from {self.path}")
        return instances

    def get_random_instance(self) -> Dict[str, Any]:
        return random.choice(self.instances)


# ---------------------------
# Geometry & TSP solver
# ---------------------------

def normalize_to_0_100(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates to [0,100] box and round to integer grid."""
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    rng = np.where((mx - mn) == 0.0, 1.0, (mx - mn))
    scaled = (coords - mn) / rng * 100.0
    return np.round(scaled).astype(int)


def euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        # vectorized distance to speed up a bit
        diff = points - points[i]
        D[i, :] = np.sqrt((diff**2).sum(axis=1))
    np.fill_diagonal(D, 0.0)
    return D

def solve_tsp_length(coords: np.ndarray) -> float:
    """Use elkai (LKH) to solve TSP and return tour length (closed tour)."""
    D = euclidean_distance_matrix(coords)
    tour = elkai.DistanceMatrix(D).solve_tsp()  # list of indices in visiting order
    # compute closed tour length
    total = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        total += D[a, b]
    return float(total)


# ---------------------------
# Sampling logic
# ---------------------------

def parse_n_arg(n_arg: str) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Parse --n argument.
    - If format like '5-30' -> returns [(5,30)], bucket_mode=False (per-n stats)
    - If comma-separated buckets '1-10,11-20,21-30' -> returns those, bucket_mode=True
    """
    parts = [p.strip() for p in n_arg.split(",")]
    ranges: List[Tuple[int, int]] = []
    bucket_mode = len(parts) > 1
    for part in parts:
        if "-" not in part:
            k = int(part)
            ranges.append((k, k))
        else:
            a, b = map(int, part.split("-"))
            if a < 2:
                a = 2  # at least 2 nodes for TSP circle to make sense
            if b < a:
                raise ValueError(f"Invalid n-range: {part}")
            ranges.append((a, b))
    return ranges, bucket_mode


def sample_subset(coords: np.ndarray, n: int) -> np.ndarray:
    if n > len(coords):
        raise ValueError(f"Requested n={n} exceeds instance size {len(coords)}")
    idx = sorted(random.sample(range(len(coords)), n))
    return coords[idx]


# ---------------------------
# Main estimation
# ---------------------------

def estimate_Ln(
    dataset_path: str,
    n_spec: str,
    samples_per_n: int,
    normalize: bool,
    seed: int,
    out_path: Optional[str]
):
    random.seed(seed)
    np.random.seed(seed)

    ranges, bucket_mode = parse_n_arg(n_spec)

    # If dataset is a directory, we will re-pick a random file each sample;
    # if it is a single file, we reuse one parser.
    dataset_p = Path(dataset_path)
    single_parser: Optional[TSPDatasetParser] = None
    if dataset_p.is_file():
        single_parser = TSPDatasetParser(str(dataset_p))

    results = {}

    if bucket_mode:
        # Bucketed mode: e.g., 1-10,11-20,21-30
        for lo, hi in ranges:
            bucket_key = f"{lo}-{hi}"
            lengths: List[float] = []
            ns: List[int] = []
            for _ in range(samples_per_n):
                # pick random n within the bucket
                n = random.randint(lo, hi)
                # pick instance file & sample subset
                if single_parser is None:
                    parser = TSPDatasetParser(get_random_dataset_file(dataset_path))
                else:
                    parser = single_parser
                inst = parser.get_random_instance()
                coords_full = inst["coordinates"]
                coords = sample_subset(coords_full, n)
                if normalize:
                    coords = normalize_to_0_100(coords)
                L = solve_tsp_length(coords)
                lengths.append(L)
                ns.append(n)

            results[bucket_key] = {
                "n_min": lo,
                "n_max": hi,
                "num_samples": samples_per_n,
                "n_mean": float(np.mean(ns)),
                "L_mean": float(np.mean(lengths)),
                "L_min": float(np.min(lengths)),
                "L_max": float(np.max(lengths)),
                "L_std": float(np.std(lengths)),
            }
            print(f"[{bucket_key}] samples={samples_per_n} | "
                  f"n̄={np.mean(ns):.2f} | L̄={np.mean(lengths):.3f} "
                  f"(min={np.min(lengths):.3f}, max={np.max(lengths):.3f}, std={np.std(lengths):.3f})")

    else:
        # Per-n mode: e.g., 5-30 -> iterate all integers n in [lo, hi]
        assert len(ranges) == 1
        lo, hi = ranges[0]
        for n in range(lo, hi + 1):
            lengths: List[float] = []
            for _ in range(samples_per_n):
                if single_parser is None:
                    parser = TSPDatasetParser(get_random_dataset_file(dataset_path))
                else:
                    parser = single_parser
                inst = parser.get_random_instance()
                coords_full = inst["coordinates"]
                coords = sample_subset(coords_full, n)
                if normalize:
                    coords = normalize_to_0_100(coords)
                L = solve_tsp_length(coords)
                lengths.append(L)
            results[str(n)] = {
                "n": n,
                "num_samples": samples_per_n,
                "L_mean": float(np.mean(lengths)),
                "L_min": float(np.min(lengths)),
                "L_max": float(np.max(lengths)),
                "L_std": float(np.std(lengths)),
            }
            print(f"[n={n:>2}] samples={samples_per_n} | L̄={np.mean(lengths):.3f} "
                  f"(min={np.min(lengths):.3f}, max={np.max(lengths):.3f}, std={np.std(lengths):.3f})")

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved stats to {out_p}")


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Estimate empirical L_n on your dataset distribution.")
    ap.add_argument("--dataset", type=str, required=True,
                    help="TSPLIB-like file or folder with such files.")
    ap.add_argument("--n", type=str, required=True,
                    help="Either a range '5-30' (per-n stats) OR comma buckets '1-10,11-20,21-30'.")
    ap.add_argument("--samples_per_n", type=int, default=50,
                    help="Samples per n (or per bucket).")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize sampled coords to [0,100] before solving.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to save JSON stats.")
    args = ap.parse_args()

    estimate_Ln(
        dataset_path=args.dataset,
        n_spec=args.n,
        samples_per_n=args.samples_per_n,
        normalize=bool(args.normalize),
        seed=args.seed,
        out_path=args.out
    )
