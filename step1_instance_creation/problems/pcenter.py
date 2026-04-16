"""
PCENTER (p-Center Problem) instance extractor.
"""
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod

from ..solvers.solvers import (
    solve_uflp_gurobi,
    solve_cflp_gurobi,
    solve_pmed_gurobi,
    solve_mdp_gurobi,
    solve_pcenter_gurobi,
)


class InstanceExtractor(ABC):
    """Abstract base class for problem instance extractors."""

    @abstractmethod
    def extract_instance(self, n_nodes: int):
        """Return (instance_dict, solution, objective_value)."""
        pass

    @abstractmethod
    def get_problem_type(self) -> str:
        pass


class PMEDDatasetParser:
    """Parser for p-median/p-center problem dataset format (shared format)."""

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instance = self._load_instance()

    def _load_instance(self) -> Dict[str, Any]:
        with open(self.path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 1:
            raise ValueError(f"Empty file: {self.path.name}")

        parts = lines[0].split()
        n_vertices = int(parts[0])
        n_edges = int(parts[1])
        p = int(parts[2])

        if len(lines) < 1 + n_edges:
            raise ValueError(
                f"Expected {n_edges} edge lines, but file has only {len(lines) - 1} lines"
            )

        edges = []
        for i in range(1, 1 + n_edges):
            parts = lines[i].split()
            v1 = int(parts[0]) - 1
            v2 = int(parts[1]) - 1
            cost = float(parts[2])
            edges.append((v1, v2, cost))

        distance_matrix = self._compute_distance_matrix(n_vertices, edges)

        return {
            'name': self.path.stem,
            'n_vertices': n_vertices,
            'p': p,
            'distance_matrix': distance_matrix,
            'edges': edges
        }

    def _compute_distance_matrix(self, n_vertices: int, edges: List[Tuple[int, int, float]]
                                 ) -> np.ndarray:
        INF = float('inf')
        dist = np.full((n_vertices, n_vertices), INF, dtype=float)

        for i in range(n_vertices):
            dist[i, i] = 0.0

        for v1, v2, cost in edges:
            dist[v1, v2] = min(dist[v1, v2], cost)
            dist[v2, v1] = min(dist[v2, v1], cost)

        for k in range(n_vertices):
            for i in range(n_vertices):
                for j in range(n_vertices):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        return dist

    def get_instance(self):
        return self.instance


class PCENTERInstanceExtractor(InstanceExtractor):
    """
    p-center instance extractor from p-median datasets.
    Uses the same data sources as p-median (distance matrices).
    """

    def __init__(
        self,
        dataset_path: str,
        time_limit: float = 60.0,
        threads: int = 8
    ):
        self.dataset_path = Path(dataset_path)
        self.is_dir = self.dataset_path.is_dir()
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "PCENTER"

    def extract_instance(self, n_vertices: int) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        if n_vertices < 1:
            raise ValueError("PCENTER requires at least 1 vertex.")

        if self.is_dir:
            files = list(self.dataset_path.glob("*.txt"))
            if not files:
                raise ValueError(f"No .txt files found in {self.dataset_path}")
            file_path = str(random.choice(files))
            print(f"Selected: {Path(file_path).name}")
            parser = PMEDDatasetParser(file_path)
        else:
            parser = PMEDDatasetParser(self.dataset_path)

        base_instance = parser.get_instance()
        distance_matrix_full = base_instance['distance_matrix']
        n_vertices_full = base_instance['n_vertices']
        p_original = base_instance['p']

        if n_vertices > n_vertices_full:
            raise ValueError(
                f"Requested {n_vertices} vertices, but instance {base_instance['name']} "
                f"only has {n_vertices_full} vertices."
            )

        if n_vertices == n_vertices_full:
            vertex_idx = list(range(n_vertices_full))
            p = p_original
        else:
            vertex_idx = sorted(random.sample(range(n_vertices_full), n_vertices))
            p = max(1, round(p_original * n_vertices / n_vertices_full))

        distance_matrix = distance_matrix_full[np.ix_(vertex_idx, vertex_idx)]

        distance_matrix = distance_matrix.astype(int)

        p = min(p, n_vertices)

        facilities, assignments, obj_value = solve_pcenter_gurobi(
            distance_matrix=distance_matrix,
            p=p,
            time_limit=self.time_limit,
            threads=self.threads
        )

        instance_dict = {
            "distance_matrix": distance_matrix.tolist(),
            "p": p,
            "objective": float(obj_value)
        }

        solution = {
            "facilities": facilities,
            "assignments": assignments
        }

        print(f"Generated PCENTER instance (n_vertices={n_vertices}, p={p}, obj={obj_value:.2f})")

        return instance_dict, solution, float(obj_value)


class InstanceGenerator:
    """Generic instance generator for PCENTER."""

    def __init__(self, extractor: InstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances):
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
                ptype = self.extractor.get_problem_type()
                if ptype in ["UFLP", "CFLP"]:
                    inst, sol, obj = self.extractor.extract_instance(n_customers=n)
                    n_facilities = len(inst['opening_costs'])
                    alpha = n / n_facilities if n_facilities > 0 else 0
                    size_str = f"n_customers={n}, n_facilities={n_facilities}, alpha={alpha:.2f}"
                elif ptype in ["PMED", "PCENTER"]:
                    inst, sol, obj = self.extractor.extract_instance(n_vertices=n)
                    p = inst['p']
                    size_str = f"n_vertices={n}, p={p}"
                elif ptype == "MDP":
                    inst, sol, obj = self.extractor.extract_instance(n_vertices=n)
                    m = inst['m']
                    size_str = f"n_vertices={n}, m={m}"
                else:
                    inst, sol, obj = self.extractor.extract_instance(n)
                    size_str = f"n={n}"

                results.append(
                    {
                        "instance": inst,
                        "solution": sol,
                        "obj": obj,
                        "problem_type": ptype,
                    }
                )
                print(f"Generated instance {len(results)}/{n_instances} ({size_str}, obj={obj:.2f})")
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
# Synthetic PCENTER Extractor (no dataset files required)
# ============================================================

class SyntheticPCENTERInstanceExtractor(InstanceExtractor):
    """Generate random p-center instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "PCENTER"

    def extract_instance(self, n_vertices: int) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        coords = np.random.randint(0, 101, size=(n_vertices, 2))
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(int)
        p = max(1, round(n_vertices * 0.25))

        facilities, assignments, obj_value = solve_pcenter_gurobi(
            distance_matrix=dist, p=p,
            time_limit=self.time_limit, threads=self.threads,
        )
        instance_dict = {"distance_matrix": dist.tolist(), "p": p, "objective": float(obj_value)}
        solution = {"facilities": facilities, "assignments": assignments}
        return instance_dict, solution, float(obj_value)
