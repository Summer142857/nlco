"""
Instance Extractor for Pickup and Delivery Problem (PDP)

- Loads instances from Li & Lim format files.
- Supports sub-sampling (samples pickup-delivery pairs together).
- Coordinates normalized to [0,100] and rounded.
- Depot is always at index 0.
- Solved with solve_pdp_gurobi() using Gurobi MILP.
"""
import ast
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..solvers.solvers import solve_pdp_gurobi


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
# PDP Dataset Parser
# ============================================================

class PDPDatasetParser:
    """
    Parser for PDP instances in the format used by Li & Lim datasets.

    File format:
    - First line: _ capacity _
    - Subsequent lines: task_no x y demand earliest latest _ pickup_idx delivery_idx

    Where:
    - Task 0 is the depot
    - Pickup tasks: demand < 0, delivery_idx = 0, pickup_idx = delivery sibling
    - Delivery tasks: demand > 0, pickup_idx = 0, delivery_idx = pickup sibling
    """

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instance = self._load_instance()

    def _load_instance(self) -> Dict[str, Any]:
        """Load and parse a PDP instance."""
        with open(self.path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        header = lines[0].split()
        capacity = int(header[1])

        coords_list = []
        demands_list = []
        time_windows_list = []
        pickup_indices = []
        delivery_indices = []

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 9:
                continue

            task_no = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
            demand = int(parts[3])
            earliest = int(parts[4])
            latest = int(parts[5])
            pickup_idx = int(parts[7])
            delivery_idx = int(parts[8])

            coords_list.append([x, y])
            demands_list.append(demand)
            time_windows_list.append([earliest, latest])
            pickup_indices.append(pickup_idx)
            delivery_indices.append(delivery_idx)

        pickup_delivery_pairs = []
        for i in range(len(demands_list)):
            if i == 0:
                continue
            if demands_list[i] < 0 and delivery_indices[i] == 0:
                pickup = pickup_indices[i]
                delivery = i
                pickup_delivery_pairs.append((pickup, delivery))

        return {
            'capacity': capacity,
            'coordinates': np.array(coords_list, dtype=int),
            'demands': np.array(demands_list, dtype=int),
            'time_windows': np.array(time_windows_list, dtype=int),
            'pickup_delivery_pairs': pickup_delivery_pairs,
        }

    def get_instance(self) -> Dict[str, Any]:
        return self.instance


# ============================================================
# PDP Instance Extractor
# ============================================================

class PDPInstanceExtractor(InstanceExtractor):
    """
    PDP extractor:
      - Loads instances from Li & Lim format files.
      - Supports sub-sampling (samples pickup-delivery pairs together).
      - Coordinates normalized to [0,100] and rounded.
      - Depot is always at index 0.
      - Solved with solve_pdp_gurobi() using Gurobi MILP.
    """

    def __init__(
        self,
        dataset_path: str,
        time_limit: float = 60.0,
        threads: int = 8,
    ):
        self.dataset_path = Path(dataset_path)
        self.is_dir = self.dataset_path.is_dir()
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "PDP"

    def extract_instance(self, n_requests: int) -> Tuple[Dict[str, Any], List[List[int]], float]:
        """
        Extract a PDP instance with n_requests pickup-delivery pairs.
        Total nodes = 1 (depot) + 2 * n_requests (pickup + delivery for each request).

        Parameters
        ----------
        n_requests : int
            Number of pickup-delivery request pairs to include.

        Returns
        -------
        instance_dict : dict
            Instance data.
        routes : List[List[int]]
            Solution routes.
        total_distance : float
            Total distance of the solution.
        """
        if n_requests < 1:
            raise ValueError("PDP requires at least 1 request (1 pickup-delivery pair).")

        if self.is_dir:
            file_path = get_random_dataset_file(self.dataset_path)
            parser = PDPDatasetParser(file_path)
        else:
            parser = PDPDatasetParser(self.dataset_path)

        base = parser.get_instance()
        coords_full = base['coordinates']
        demands_full = base['demands']
        time_windows_full = base['time_windows']
        pairs_full = base['pickup_delivery_pairs']
        capacity = base['capacity']

        n_pairs_full = len(pairs_full)
        if n_requests > n_pairs_full:
            raise ValueError(
                f"Requested {n_requests} pairs, but instance only has {n_pairs_full} pairs."
            )

        if n_requests == n_pairs_full:
            sampled_pairs = pairs_full
        else:
            sampled_pairs = random.sample(pairs_full, n_requests)

        sampled_indices = [0]
        for pickup, delivery in sampled_pairs:
            sampled_indices.append(pickup)
            sampled_indices.append(delivery)

        sampled_indices = sorted(sampled_indices)

        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}

        coords = coords_full[sampled_indices]
        demands = demands_full[sampled_indices]
        time_windows = time_windows_full[sampled_indices]

        new_pairs = []
        for pickup, delivery in sampled_pairs:
            new_pairs.append((old_to_new[pickup], old_to_new[delivery]))

        coords = self._normalize(coords)

        depot = 0

        routes, total_dist = solve_pdp_gurobi(
            coords=coords,
            demands=demands,
            time_windows=time_windows,
            pickup_delivery_pairs=new_pairs,
            depot=depot,
            time_limit=self.time_limit,
            threads=self.threads,
        )

        instance_dict = {
            "coordinates": coords.tolist(),
            "depot": depot,
            "time_windows": time_windows.tolist(),
            "pickup_delivery_pairs": new_pairs,
            "num_vehicles": len(routes),
            "total_distance": float(total_dist),
            "objective": float(total_dist),
        }

        print(f"Generated PDP instance ({n_requests} requests, {len(routes)} vehicles, dist={total_dist:.2f})")

        return instance_dict, routes, float(total_dist)

    def _normalize(self, coordinates: np.ndarray) -> np.ndarray:
        """Scale coordinates to [0,100] and round to integers."""
        min_vals = coordinates.min(axis=0)
        max_vals = coordinates.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        normalized = (coordinates - min_vals) / range_vals * 100
        return np.round(normalized).astype(int)


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
# Synthetic PDP Extractor (no dataset files required)
# ============================================================

class SyntheticPDPInstanceExtractor(InstanceExtractor):
    """Generate random pickup-delivery instances synthetically."""

    def __init__(self, capacity: int = 100, time_limit: float = 60.0, threads: int = 8):
        self.capacity = capacity
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "PDP"

    def extract_instance(self, n_requests: int) -> Tuple[Dict[str, Any], List[List[int]], float]:
        if n_requests < 1:
            raise ValueError("PDP requires at least 1 request.")
        n_nodes = 1 + 2 * n_requests  # depot + pickup + delivery per request
        coords = np.random.randint(0, 101, size=(n_nodes, 2))
        demands = np.zeros(n_nodes, dtype=int)
        time_windows = np.zeros((n_nodes, 2), dtype=int)
        time_windows[:, 1] = 1000  # large time window
        pairs = []
        for i in range(n_requests):
            pickup_idx = 1 + 2 * i
            delivery_idx = 1 + 2 * i + 1
            demands[pickup_idx] = random.randint(5, 20)
            demands[delivery_idx] = -demands[pickup_idx]
            pairs.append((pickup_idx, delivery_idx))

        routes, total_dist = solve_pdp_gurobi(
            coords=coords,
            demands=demands,
            time_windows=time_windows,
            pickup_delivery_pairs=pairs,
            depot=0,
            time_limit=self.time_limit,
            threads=self.threads,
        )
        instance_dict = {
            "coordinates": coords.tolist(),
            "depot": 0,
            "time_windows": time_windows.tolist(),
            "pickup_delivery_pairs": pairs,
            "num_vehicles": len(routes),
            "total_distance": float(total_dist),
            "objective": float(total_dist),
        }
        return instance_dict, routes, float(total_dist)
