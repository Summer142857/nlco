"""
MDP (Maximum Diversity Problem) instance extractor.
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


class MDPDatasetParser:
    """Parser for Maximum Diversity Problem (MDP) dataset format."""

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
        m = int(parts[1])

        distance_matrix = np.zeros((n_vertices, n_vertices), dtype=float)

        for i in range(1, len(lines)):
            parts = lines[i].split()
            if len(parts) >= 3:
                v1 = int(parts[0])
                v2 = int(parts[1])
                dist = float(parts[2])

                distance_matrix[v1, v2] = dist
                distance_matrix[v2, v1] = dist

        distance_matrix = distance_matrix.astype(int)

        return {
            'name': self.path.stem,
            'n_vertices': n_vertices,
            'm': m,
            'distance_matrix': distance_matrix
        }

    def get_instance(self):
        return self.instance


class MDPInstanceExtractor(InstanceExtractor):
    """
    MDP instance extractor from MDP datasets.
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
        return "MDP"

    def extract_instance(self, n_vertices: int) -> Tuple[Dict[str, Any], List[int], float]:
        if n_vertices < 2:
            raise ValueError("MDP requires at least 2 vertices.")

        if self.is_dir:
            files = list(self.dataset_path.glob("*.txt"))
            if not files:
                raise ValueError(f"No .txt files found in {self.dataset_path}")
            file_path = str(random.choice(files))
            print(f"Selected: {Path(file_path).name}")
            parser = MDPDatasetParser(file_path)
        else:
            parser = MDPDatasetParser(self.dataset_path)

        base_instance = parser.get_instance()
        distance_matrix_full = base_instance['distance_matrix']
        n_vertices_full = base_instance['n_vertices']
        m_original = base_instance['m']

        if n_vertices > n_vertices_full:
            raise ValueError(
                f"Requested {n_vertices} vertices, but instance {base_instance['name']} "
                f"only has {n_vertices_full} vertices."
            )

        if n_vertices == n_vertices_full:
            vertex_idx = list(range(n_vertices_full))
            m = m_original
        else:
            vertex_idx = sorted(random.sample(range(n_vertices_full), n_vertices))
            m = max(2, round(m_original * n_vertices / n_vertices_full))

        distance_matrix = distance_matrix_full[np.ix_(vertex_idx, vertex_idx)]

        m = min(m, n_vertices)

        selected_elements, obj_value = solve_mdp_gurobi(
            distance_matrix=distance_matrix,
            m=m,
            time_limit=self.time_limit,
            threads=self.threads
        )

        instance_dict = {
            "distance_matrix": distance_matrix.tolist(),
            "m": m,
            "objective": float(obj_value)
        }

        print(f"Generated MDP instance (n_vertices={n_vertices}, m={m}, obj={obj_value:.2f})")

        return instance_dict, selected_elements, float(obj_value)


class InstanceGenerator:
    """Generic instance generator for MDP."""

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
# Synthetic MDP Extractor (no dataset files required)
# ============================================================

class SyntheticMDPInstanceExtractor(InstanceExtractor):
    """Generate random MDP instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "MDP"

    def extract_instance(self, n_vertices: int) -> Tuple[Dict[str, Any], List[int], float]:
        coords = np.random.rand(n_vertices, 2) * 100
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(int)
        m = max(2, round(n_vertices * 0.4))

        selected_elements, obj_value = solve_mdp_gurobi(
            distance_matrix=dist, m=m,
            time_limit=self.time_limit, threads=self.threads,
        )
        instance_dict = {"distance_matrix": dist.tolist(), "m": m, "objective": float(obj_value)}
        return instance_dict, selected_elements, float(obj_value)
