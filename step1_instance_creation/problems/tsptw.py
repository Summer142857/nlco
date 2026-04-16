"""
TSPTW (Traveling Salesman Problem with Time Windows) instance extractor.

Samples nodes from TSPLIB-format datasets, generates random time windows
calibrated to expected tour length, and solves with LKH-3 (Gurobi fallback).
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .tsp import TSPDatasetParser
from ..utils.io import get_random_dataset_file
from ..solvers.solvers import solve_tsptw_lkh


class TSPTWInstanceExtractor:
    """
    TSPTW instance extractor using TSPLIB-style coordinate datasets.

    Time windows follow the Easy TSPTW convention:
        l_i ~ U[0, T_N]
        u_i = l_i + T_N * U[alpha, beta]
    where T_N approximates the expected TSP tour length.
    """

    def __init__(
        self,
        dataset_path: str,
        alpha: float = 0.5,
        beta: float = 0.75,
        time_limit: float = 60.0,
        threads: int = 8,
    ):
        self.dataset_path = dataset_path
        self.is_dir = Path(dataset_path).is_dir()
        self.parser = None if self.is_dir else TSPDatasetParser(dataset_path)
        self.alpha = alpha
        self.beta = beta
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "TSPTW"

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        if self.is_dir:
            file_path = get_random_dataset_file(self.dataset_path)
            parser = TSPDatasetParser(file_path)
        else:
            parser = self.parser

        base = parser.get_random_instance()
        coords_full = base["coordinates"]

        if n_nodes < 2:
            raise ValueError("TSPTW requires at least 2 nodes (including depot).")
        if n_nodes > len(coords_full):
            raise ValueError(
                f"Requested {n_nodes} nodes, but only {len(coords_full)} available."
            )

        idx = sorted(random.sample(range(len(coords_full)), n_nodes))
        coords = coords_full[idx]
        coords = self._normalize(coords)

        n = coords.shape[0]
        depot = 0

        # calibrated expected TSP tour length
        if n <= 10:
            T_N = 334
        elif n <= 20:
            T_N = 388
        else:
            T_N = 451

        alpha = self.alpha
        beta = self.beta

        l = np.zeros(n, dtype=float)
        u = np.zeros(n, dtype=float)

        l[depot] = 0.0
        u[depot] = float((2.0 + beta) * T_N)

        for i in range(n):
            if i == depot:
                continue
            l_i = np.random.uniform(0.0, T_N)
            width = T_N * np.random.uniform(alpha, beta)
            l[i] = int(l_i)
            u[i] = int(l_i + width)

        time_windows = np.vstack([l, u]).T  # shape (n, 2)

        tour, total_dist = solve_tsptw_lkh(
            coords=coords,
            time_windows=time_windows,
            depot=depot,
            time_limit=self.time_limit,
            threads=self.threads,
        )

        if len(tour) != n + 1:
            raise ValueError(f"Tour length {len(tour)} != number of nodes {n}")

        time_windows = time_windows.astype(int)
        instance_dict = {
            "coordinates": coords.tolist(),
            "depot": depot,
            "num_nodes": n,
            "time_windows": time_windows.tolist(),
            "tour_length": float(total_dist),
            "objective": float(total_dist),
        }
        return instance_dict, tour, float(total_dist)

    def _normalize(self, coordinates: np.ndarray) -> np.ndarray:
        """Normalize coordinates to [0, 100] and round to integers."""
        min_vals = coordinates.min(axis=0)
        max_vals = coordinates.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        normalized = (coordinates - min_vals) / range_vals * 100.0
        return np.round(normalized).astype(int)


class InstanceGenerator:
    """Generate and save TSPTW instances."""

    def __init__(self, extractor: TSPTWInstanceExtractor):
        self.extractor = extractor

    def generate_instances(
        self,
        n_nodes: Union[int, Tuple[int, int]],
        n_instances: int,
    ) -> List[Dict[str, Any]]:
        instances: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = n_instances * 10

        while len(instances) < n_instances:
            attempts += 1
            if attempts > max_attempts:
                print(f"Warning: max attempts reached. Generated {len(instances)}/{n_instances}.")
                break
            try:
                current_n = random.randint(*n_nodes) if isinstance(n_nodes, tuple) else n_nodes
                inst, sol, obj = self.extractor.extract_instance(current_n)
                instances.append({
                    "instance": inst,
                    "solution": sol,
                    "obj": obj,
                    "problem_type": "TSPTW",
                })
                if len(instances) % 10 == 0 or len(instances) == n_instances:
                    print(f"Generated {len(instances)}/{n_instances} TSPTW instances")
            except Exception as e:
                print(f"Error generating TSPTW instance: {e}")
                continue

        return instances

    def generate_and_save(
        self,
        n_nodes: Union[int, Tuple[int, int]],
        n_instances: int,
        out_path: str = None,
        output_path: str = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        path = out_path if out_path is not None else output_path
        if path is None:
            raise ValueError("Must provide out_path or output_path")
        if isinstance(n_nodes, tuple):
            print(f"Generating {n_instances} TSPTW instances with {n_nodes[0]}-{n_nodes[1]} nodes...")
        else:
            print(f"Generating {n_instances} TSPTW instances with {n_nodes} nodes...")
        instances = self.generate_instances(n_nodes, n_instances)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"Saved {len(instances)} TSPTW instances to {out}")
        return instances
