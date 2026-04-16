"""
Instance Extractor for Prize-Collecting Traveling Salesman Problem (PCTSP)

- Reads coordinates from TSPLIB-like text files (one Python-list per line).
- Adds prizes/penalties and computes a minimum required prize.
- Solves the PCTSP exactly with Gurobi (MILP, MTZ subtour constraints with optional nodes).
"""
import ast
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

import gurobipy as gp
from gurobipy import GRB
from ..solvers.solvers import solve_pctsp_gurobi, solve_pctsp_hgs


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
# Parser for TSPLIB-like data
# ============================================================

class TSPDatasetParser:
    """Minimal parser for TSPLIB-like datasets."""

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instances = self._load_instances()

    def _load_instances(self) -> List[Dict[str, Any]]:
        """Load and parse all instances from the dataset file."""
        instances = []

        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = ast.literal_eval(line)
                    instance_name = data[0]
                    optimal_value = float(data[1])

                    coords_flat = [float(x) for x in data[2:]]

                    coordinates = []
                    for i in range(0, len(coords_flat), 2):
                        coordinates.append([coords_flat[i], coords_flat[i + 1]])

                    instances.append({
                        'name': instance_name,
                        'optimal_value': optimal_value,
                        'coordinates': np.array(coordinates)
                    })

        return instances

    def get_random_instance(self):
        return random.choice(self.instances)


# ============================================================
# PCTSP Extractor
# ============================================================

class PCTSPInstanceExtractor(InstanceExtractor):
    """Prize-Collecting TSP extractor built on TSPLIB coordinates + Gurobi solver."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.is_dir = Path(dataset_path).is_dir()
        self.parser = None if self.is_dir else TSPDatasetParser(dataset_path)

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        """Extract a random PCTSP instance and solve it with Gurobi."""
        if self.is_dir:
            file_path = get_random_dataset_file(self.dataset_path)
            parser = TSPDatasetParser(file_path)
        else:
            parser = self.parser

        inst = parser.get_random_instance()
        coords_full = inst["coordinates"]

        if n_nodes < 2:
            raise ValueError("PCTSP requires at least 2 nodes (including depot).")
        if n_nodes > len(coords_full):
            raise ValueError(f"Requested {n_nodes} nodes, but only {len(coords_full)} available")

        idx = sorted(random.sample(range(len(coords_full)), n_nodes))
        coords = coords_full[idx]
        coords = self._normalize(coords)

        n = coords.shape[0]
        depot = 0

        prizes = np.round(np.random.uniform(0.0, 100.0, size=n)).astype(int)

        if n <= 10:
            L_n = 334
        elif n <= 20:
            L_n = 388
        else:
            L_n = 451

        penalties = np.round(np.random.uniform(0, 3 * L_n / (2 * n), size=n)).astype(int)
        required_prize = int(100 * n / 4.0)
        prizes[depot] = 0.0
        penalties[depot] = 0.0

        tour, obj = solve_pctsp_gurobi(coords, prizes, penalties, required_prize, depot=depot)

        route_length = sum(
            np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]])
            for i in range(len(tour) - 1)
        )

        route_prize = prizes[tour].sum()

        visited = set(tour)
        unvisited_penalty = sum(penalties[i] for i in range(n) if i not in visited)

        instance_dict = {
            "coordinates": coords.tolist(),
            "prizes": prizes.tolist(),
            "penalties": penalties.tolist(),
            "required_prize": float(required_prize),
            "depot": depot,
            "route_length": float(route_length),
            "route_prize": float(route_prize),
            "unvisited_penalty": float(unvisited_penalty),
            "objective": float(obj)
        }

        return instance_dict, tour, float(obj)

    def _normalize(self, coordinates: np.ndarray) -> np.ndarray:
        """Normalize coordinates to [0, 100] range and round to integers."""
        min_vals = coordinates.min(axis=0)
        max_vals = coordinates.max(axis=0)

        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)

        normalized = (coordinates - min_vals) / range_vals * 100
        normalized = np.round(normalized).astype(int)

        return normalized

    def get_problem_type(self) -> str:
        return "PCTSP"


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
