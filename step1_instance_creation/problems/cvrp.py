"""
Instance Extractor for Capacitated Vehicle Routing Problem (CVRP)

- Reads VRPLIB-style instances from text files.
- Solves with Gurobi MILP.
"""
import ast
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..solvers.solvers import solve_cvrp_gurobi


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
# CVRP Instance Extractor (VRPLIB-based)
# ============================================================

def _parse_vrp_file(path: Path) -> Dict[str, Any]:
    """Parse a standard VRPLIB .vrp file (EUC_2D format)."""
    coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}
    capacity = 1
    name = path.stem
    section = None

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("NAME"):
                name = line.split(":", 1)[-1].strip()
            elif upper.startswith("CAPACITY"):
                capacity = int(line.split(":", 1)[-1].strip())
            elif upper.startswith("NODE_COORD_SECTION"):
                section = "coord"
            elif upper.startswith("DEMAND_SECTION"):
                section = "demand"
            elif upper.startswith("DEPOT_SECTION"):
                section = "depot"
            elif upper.startswith("EOF") or upper.startswith("END"):
                break
            elif section == "coord":
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    coords[node_id] = (float(parts[1]), float(parts[2]))
            elif section == "demand":
                parts = line.split()
                if len(parts) >= 2:
                    demands[int(parts[0])] = int(parts[1])

    if not coords:
        raise ValueError(f"No coordinates found in {path}")

    sorted_ids = sorted(coords.keys())
    n = len(sorted_ids)
    coord_arr = np.zeros((n, 2), dtype=float)
    demand_arr = np.zeros(n, dtype=int)
    for i, node_id in enumerate(sorted_ids):
        coord_arr[i] = coords[node_id]
        demand_arr[i] = demands.get(node_id, 0)

    return {
        "name": name,
        "coordinates": coord_arr,
        "demands": demand_arr,
        "capacity": capacity,
        "best_cost": 0.0,
    }


class CVRPInstanceExtractor(InstanceExtractor):

    def __init__(self, dataset_path: str,
                 time_limit: float = 60.0,
                 threads: int = 8):
        """
        Parameters
        ----------
        dataset_path : str
            Path to a directory of .vrp files OR a VRPLIB-style text file
            where each line is a Python list literal.
        """
        self.dataset_path = Path(dataset_path)
        self.instances = self._load_instances()
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "CVRP"

    def _load_instances(self) -> List[Dict[str, Any]]:
        """Load CVRP instances — supports .vrp directory or Python-list text file."""
        if self.dataset_path.is_dir():
            vrp_files = list(self.dataset_path.glob("*.vrp"))
            if not vrp_files:
                raise ValueError(f"No .vrp files found in {self.dataset_path}")
            instances = []
            for f in sorted(vrp_files):
                try:
                    instances.append(_parse_vrp_file(f))
                except Exception as e:
                    print(f"[WARN] Could not parse {f.name}: {e}")
            if not instances:
                raise ValueError(f"No CVRP instances loaded from {self.dataset_path}")
            return instances

        # Fall back: Python-list-per-line text file
        instances: List[Dict[str, Any]] = []
        with open(self.dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = ast.literal_eval(line)
                inst = self._parse_vrplib_row(row)
                instances.append(inst)
        if not instances:
            raise ValueError(f"No CVRP instances found in {self.dataset_path}")
        return instances

    def _parse_vrplib_row(self, row: List[Any]) -> Dict[str, Any]:
        """
        Parse a single VRPLIB-style row into coordinates, demands, and capacity.
        """
        i = 0
        if row[i] != "name":
            raise ValueError("Expected 'name' token at start of row.")
        name = row[i + 1]
        i += 2

        if row[i] != "depot":
            raise ValueError("Expected 'depot' token after name.")
        depot_x = float(row[i + 1])
        depot_y = float(row[i + 2])
        i += 3

        if row[i] != "customer":
            raise ValueError("Expected 'customer' token after depot coordinates.")
        i += 1

        customer_coords: List[Tuple[float, float]] = []
        while i < len(row) and row[i] != "demand":
            x = float(row[i])
            y = float(row[i + 1])
            customer_coords.append((x, y))
            i += 2

        if i >= len(row) or row[i] != "demand":
            raise ValueError("Expected 'demand' token in VRPLIB row.")
        i += 1

        demands: List[int] = []
        while i < len(row) and row[i] != "capacity":
            demands.append(int(row[i]))
            i += 1

        if i >= len(row) or row[i] != "capacity":
            raise ValueError("Expected 'capacity' token in VRPLIB row.")
        capacity = int(row[i + 1])
        i += 2

        if i >= len(row) or row[i] != "cost":
            raise ValueError("Expected 'cost' token in VRPLIB row.")
        best_cost = float(row[i + 1])

        n = 1 + len(customer_coords)
        coords = np.zeros((n, 2), dtype=float)
        coords[0, 0] = depot_x
        coords[0, 1] = depot_y
        for j, (x, y) in enumerate(customer_coords, start=1):
            coords[j, 0] = x
            coords[j, 1] = y

        if len(demands) != n:
            raise ValueError(
                f"Mismatch between number of coordinates ({n}) and demands ({len(demands)}) "
                f"for instance {name}."
            )

        return {
            "name": name,
            "coordinates": coords,
            "demands": np.array(demands, dtype=int),
            "capacity": capacity,
            "best_cost": best_cost,
        }

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[List[int]], float]:
        """
        n_nodes = depot + customers.
        """
        if n_nodes < 2:
            raise ValueError("CVRP requires at least 2 nodes (1 depot + >=1 customer).")

        base = random.choice(self.instances)
        coords_full: np.ndarray = base["coordinates"]
        demands_full: np.ndarray = base["demands"]
        capacity_full: int = int(base["capacity"])

        n_total = coords_full.shape[0]
        if n_nodes > n_total:
            raise ValueError(
                f"Requested {n_nodes} nodes, but VRPLIB instance "
                f"{base['name']} only has {n_total} nodes (including depot)."
            )

        if n_nodes == n_total:
            idx = list(range(n_total))
        else:
            customer_indices = list(range(1, n_total))
            sampled_customers = sorted(random.sample(customer_indices, n_nodes - 1))
            idx = [0] + sampled_customers

        coords = coords_full[idx]
        demands = demands_full[idx]

        coords = self._normalize(coords)

        n = coords.shape[0]
        depot = 0

        capacity = capacity_full

        routes, total_dist = solve_cvrp_gurobi(
            coords=coords,
            demands=demands,
            capacity=capacity,
            depot=depot,
            time_limit=self.time_limit,
            threads=self.threads,
        )

        instance_dict = {
            "coordinates": coords.tolist(),
            "depot": depot,
            "demands": demands.tolist(),
            "capacity": int(capacity),
            "num_vehicles": len(routes),
            "total_distance": float(total_dist),
            "objective": float(total_dist),
        }
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
