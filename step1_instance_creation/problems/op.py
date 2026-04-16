"""
Instance Extractor for Orienteering Problem (OP)

- Reads Tsiligirides-style OP instances from text files.
- Solves them using Gurobi MIP (small) or COMPASS heuristic (large).
"""
import ast
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..solvers.solvers import solve_op_gurobi_mip, solve_op_compass


# ============================================================
# Utility functions
# ============================================================

def get_random_dataset_file(dataset_path: str) -> str:
    """Select a random dataset file."""
    dataset_path = Path(dataset_path)
    if dataset_path.is_file():
        return str(dataset_path)
    elif dataset_path.is_dir():
        files = []
        for pattern in ["*.tsp", "*.txt", "*.dat"]:
            files.extend(dataset_path.glob(pattern))
        if not files:
            raise ValueError(f"No dataset files found in {dataset_path}")
        selected = random.choice(files)
        print(f"{selected}")
        return str(selected)
    else:
        raise ValueError(f"Invalid dataset path: {dataset_path}")


# ============================================================
# Base Class
# ============================================================

class InstanceExtractor(ABC):
    """Abstract base class for problem instance extractors."""

    @abstractmethod
    def extract_instance(self, n_nodes: int) -> Tuple[Any, Any, float]:
        """Return (instance_dict, solution, objective_value)."""
        pass

    @abstractmethod
    def get_problem_type(self) -> str:
        pass


# ============================================================
# OP Instance Extractor (Tsiligirides + COMPASS solver)
# ============================================================

class OPInstanceExtractor(InstanceExtractor):
    """
    OP extractor that can read Tsiligirides-style OP instances from text files
    and solve them using the external COMPASS solver (via solve_op_gurobi_mip
    or solve_op_compass in solvers.py).

    Expected file format for each instance (under ``datasets/OP/``):

        - First line: ``Tmax P``
              Tmax = available time budget per path
              P    = number of paths (we assume P = 1)

        - Remaining lines: one point per line, with:
              x  y  S
              x, y = coordinates (floats)
              S    = score (prize)

        Remarks:
          - The first point is the starting point.
          - The second point is the ending point.
          - Euclidean distance is used.
    """

    def __init__(
        self,
        dataset_path: str,
        time_limit: float = 60.0,
        threads: int = 8,
    ):
        self.dataset_path = Path(dataset_path)
        if self.dataset_path.is_dir():
            self.files = sorted(self.dataset_path.glob("*.txt"))
        else:
            self.files = [self.dataset_path]

        if not self.files:
            raise ValueError(f"No OP instance files found at {self.dataset_path}")

        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "OP"

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        """
        Extract a random OP instance, optionally sub-sampling nodes.

        n_nodes counts all nodes (including start and end). We always keep
        the first two nodes (start and end) from the source file when
        sub-sampling.
        """
        if n_nodes < 2:
            raise ValueError("OP requires at least 2 nodes (including the depot).")

        file_path = random.choice(self.files)
        _, coords_full, prizes_full = self._load_op_file(file_path)

        orig_total = coords_full.shape[0]
        if orig_total <= 2:
            raise ValueError(
                f"OP instance {file_path.name} must have at least 3 points "
                f"(origin, old destination, and at least one customer)."
            )
        keep_indices = [0] + list(range(2, orig_total))
        coords_full = coords_full[keep_indices]
        prizes_full = prizes_full[keep_indices]

        Tmax = random.randint(5, 30)

        total_nodes = coords_full.shape[0]
        if n_nodes > total_nodes:
            raise ValueError(
                f"Requested {n_nodes} nodes, but OP instance {file_path.name} "
                f"only has {total_nodes} points."
            )

        if n_nodes == total_nodes:
            idx = list(range(total_nodes))
        elif n_nodes == 1:
            idx = [0]
        else:
            remaining = list(range(1, total_nodes))
            need = n_nodes - 1
            if need > len(remaining):
                raise ValueError(
                    f"Cannot select {n_nodes} nodes from OP instance {file_path.name} "
                    f"while keeping the origin."
                )
            sampled = sorted(random.sample(remaining, need))
            idx = [0] + sampled

        coords = coords_full[idx]
        prizes = prizes_full[idx]

        n = coords.shape[0]
        depot = 0

        prizes[depot] = 0.0

        if n <= 50:
            tour, collected, travel_len = solve_op_gurobi_mip(
                coords=coords,
                prizes=prizes,
                max_length=Tmax,
                depot=depot,
                time_limit=self.time_limit,
                threads=self.threads,
            )
        else:
            tour, collected, travel_len = solve_op_compass(
                coords=coords,
                prizes=prizes,
                max_length=Tmax,
                depot=depot,
                time_limit=self.time_limit,
                threads=self.threads,
            )

        if collected <= 0.0:
            raise ValueError(f"Collected prize is 0.0 for instance {file_path.name}")

        instance_dict = {
            "coordinates": coords.tolist(),
            "prizes": prizes.tolist(),
            "depot": depot,
            "max_length": float(Tmax),
            "route_length": float(travel_len),
            "collected_prize": float(collected),
            "objective": float(collected),
        }
        return instance_dict, tour, float(collected)

    def _load_op_file(
        self, path: Path
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Load a Tsiligirides-style OP instance from a single file.

        Returns
        -------
        Tmax : float
            Available time budget per path.
        coords : np.ndarray
            Array of shape (N, 2) with (x, y) coordinates.
        prizes : np.ndarray
            Array of shape (N,) with scores S.
        """
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        first = lines[0].split()
        if len(first) < 2:
            raise ValueError(f"Invalid header line in OP file {path}: {lines[0]}")

        Tmax = float(first[0])

        coords_list: List[Tuple[int, int]] = []
        prizes_list: List[int] = []

        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid point line in OP file {path}: {ln}")
            x = int(float(parts[0]))
            y = int(float(parts[1]))
            s = int(float(parts[2]))
            coords_list.append((x, y))
            prizes_list.append(s)

        if not coords_list:
            raise ValueError(f"No points found in OP file {path}")

        coords = np.array(coords_list, dtype=int)
        prizes = np.array(prizes_list, dtype=int)

        return Tmax, coords, prizes


# ============================================================
# Generator Class
# ============================================================

class InstanceGenerator:
    """Generic instance generator."""

    def __init__(self, extractor: InstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances):
        """
        Generate exactly n_instances successful instances, retrying on errors.
        """
        results = []
        attempts = 0
        max_attempts = n_instances * 10

        while len(results) < n_instances and attempts < max_attempts:
            attempts += 1
            if isinstance(n_nodes, tuple):
                n = random.randint(*n_nodes)
            else:
                n = n_nodes
            try:
                inst, sol, obj = self.extractor.extract_instance(n)
                results.append(
                    {
                        "instance": inst,
                        "solution": sol,
                        "obj": obj,
                        "problem_type": self.extractor.get_problem_type(),
                    }
                )
            except Exception as e:
                print(f"Error generating instance (attempt {attempts}): {e}")
                continue

        if len(results) < n_instances:
            print(
                f"Warning: requested {n_instances} instances but only "
                f"generated {len(results)} after {attempts} attempts."
            )

        return results

    def save_to_json(self, instances, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"Saved {len(instances)} instances to {out_path}")

    def generate_and_save(self, n_nodes, n_instances, out_path, **kwargs):
        data = self.generate_instances(n_nodes, n_instances)
        self.save_to_json(data, out_path)
        return data


# ============================================================
# Synthetic OP Extractor (no dataset files required)
# ============================================================

class SyntheticOPInstanceExtractor(InstanceExtractor):
    """Generate random Orienteering Problem instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "OP"

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        if n_nodes < 2:
            raise ValueError("OP requires at least 2 nodes.")
        coords = np.random.randint(0, 101, size=(n_nodes, 2))
        prizes = np.random.randint(1, 20, size=n_nodes)
        prizes[0] = 0  # depot has no prize
        # Tmax: roughly 30-50% of full tour length
        dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        Tmax = max(10.0, float(dists.mean() * n_nodes * 0.4))

        tour, collected, travel_len = solve_op_gurobi_mip(
            coords=coords,
            prizes=prizes,
            max_length=Tmax,
            depot=0,
            time_limit=self.time_limit,
            threads=self.threads,
        )
        instance_dict = {
            "coordinates": coords.tolist(),
            "prizes": prizes.tolist(),
            "max_length": float(Tmax),
            "depot": 0,
            "collected_prize": float(collected),
            "travel_length": float(travel_len),
            "objective": float(collected),
        }
        return instance_dict, tour, float(collected)
