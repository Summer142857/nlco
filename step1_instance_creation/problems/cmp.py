"""
Instance Extractor for Cutwidth Minimization Problem (CMP)

- Reads graph instances from CMP dataset files (one graph per file).
- Supports sub-sampling vertices: builds an induced subgraph on n_nodes vertices.
- Solves Cutwidth Minimization with Gurobi (solve_cmp_gurobi).
- Returns a permutation of vertices (linear layout) and its cutwidth.
"""
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..solvers.solvers import solve_cmp_gurobi


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
# CMP Dataset Parser
# ============================================================

class CMPDatasetParser:
    """
    Parser for CMP (Cutwidth Minimization Problem) instances.

    Expected format (example):

        Nombre del problema: p17_16_24
        16 16 24
        1 16
        2 14
        ...
        4 12

    We interpret:
      - First line: problem name.
      - Second line: three integers, we use:
            n = first int
            m = last int  (number of edges)
      - Remaining lines: m edges "u v" with 1-based vertex indices.
    """

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instance = self._load_instance()

    def _load_instance(self) -> Dict[str, Any]:
        with open(self.path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            raise ValueError(f"Empty CMP instance file: {self.path}")

        first = lines[0]
        if ":" in first:
            name = first.split(":", 1)[1].strip()
        else:
            name = first.strip()

        if len(lines) < 2:
            raise ValueError(f"Missing header line in CMP file {self.path}")
        header_parts = lines[1].split()
        if len(header_parts) < 2:
            raise ValueError(f"Invalid header line in CMP file {self.path}: {lines[1]}")

        header_ints = [int(x) for x in header_parts]
        n = header_ints[0]
        m = header_ints[-1]

        edges: List[Tuple[int, int]] = []
        for ln in lines[2:]:
            parts = ln.split()
            if len(parts) < 2:
                continue
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            edges.append((u, v))

        if len(edges) != m:
            print(
                f"Warning: CMP file {self.path} header says m={m} edges, "
                f"but {len(edges)} were read."
            )

        max_idx = -1
        if edges:
            max_idx = max(max(u, v) for (u, v) in edges)
        n_inferred = max_idx + 1 if max_idx >= 0 else n
        if n_inferred > n:
            n = n_inferred

        return {
            "name": name,
            "n": n,
            "m": len(edges),
            "edges": edges,
        }

    def get_instance(self) -> Dict[str, Any]:
        return self.instance


# ============================================================
# CMP Instance Extractor
# ============================================================

class CMPInstanceExtractor(InstanceExtractor):
    """
    CMP extractor:
      - Reads graph instances from CMP dataset files (one graph per file).
      - Supports sub-sampling vertices: builds an induced subgraph on n_nodes vertices.
      - Solves Cutwidth Minimization with Gurobi (solve_cmp_gurobi).
      - Returns a permutation of vertices (linear layout) and its cutwidth.
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

        if self.is_dir:
            files = []
            for file in self.dataset_path.glob("*"):
                if file.is_file():
                    files.append(file)
            if not files:
                raise ValueError(f"No CMP instance files found in {self.dataset_path}")
            self.files = sorted(files)
        else:
            if not self.dataset_path.exists():
                raise ValueError(f"CMP dataset file not found: {self.dataset_path}")
            self.files = [self.dataset_path]

    def get_problem_type(self) -> str:
        return "CMP"

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        """
        Extract a CMP instance with n_nodes vertices (induced subgraph).

        - If n_nodes equals the graph size, we use the full instance.
        - If n_nodes is smaller, we sample a subset of vertices and take
          the induced subgraph, making sure there is at least one edge.
        """
        if n_nodes < 2:
            raise ValueError("CMP requires at least 2 nodes.")

        file_path = random.choice(self.files)
        parser = CMPDatasetParser(str(file_path))
        base = parser.get_instance()

        n_full: int = base["n"]
        edges_full: List[Tuple[int, int]] = base["edges"]

        if n_nodes > n_full:
            raise ValueError(
                f"Requested {n_nodes} nodes, but CMP instance {base['name']} "
                f"only has {n_full} vertices."
            )

        if not edges_full:
            raise ValueError(
                f"CMP instance {base['name']} has no edges in the full graph."
            )

        if n_nodes == n_full:
            selected_old = list(range(n_full))
        else:
            u0, v0 = random.choice(edges_full)
            selected_set = {u0, v0}

            remaining_vertices = [i for i in range(n_full) if i not in selected_set]
            need = n_nodes - len(selected_set)
            if need > 0:
                extra = random.sample(remaining_vertices, need)
                selected_set.update(extra)

            selected_old = sorted(selected_set)

        old_to_new = {old: new for new, old in enumerate(selected_old)}
        n = len(selected_old)

        edges_sub: List[Tuple[int, int]] = []
        for (u, v) in edges_full:
            if u in old_to_new and v in old_to_new:
                edges_sub.append((old_to_new[u], old_to_new[v]))

        if not edges_sub:
            raise ValueError(
                f"Induced subgraph for instance {base['name']} has no edges "
                f"(this should not happen if edges_full was non-empty)."
            )

        layout, cutwidth = solve_cmp_gurobi(
            n=n,
            edges=edges_sub,
            time_limit=self.time_limit,
            threads=self.threads,
        )

        if n == n_full:
            inst_name = base["name"]
        else:
            inst_name = f"{base['name']}_sub_{n}"

        instance_dict = {
            "name": inst_name,
            "num_nodes": n,
            "num_edges": len(edges_sub),
            "edges": [[int(u), int(v)] for (u, v) in edges_sub],
            "solution": [int(i) for i in layout],
            "objective": float(cutwidth),
        }

        return instance_dict, layout, float(cutwidth)


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
# Synthetic CMP Extractor (no dataset files required)
# ============================================================

class SyntheticCMPInstanceExtractor(InstanceExtractor):
    """Generate random Erdős-Rényi graphs for CMP."""

    def __init__(self, edge_prob: float = 0.3, time_limit: float = 60.0, threads: int = 8):
        self.edge_prob = edge_prob
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "CMP"

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        import numpy as np
        if n_nodes < 2:
            raise ValueError("CMP requires at least 2 nodes.")
        edges: List[Tuple[int, int]] = []
        for u in range(n_nodes):
            for v in range(u + 1, n_nodes):
                if random.random() < self.edge_prob:
                    edges.append((u, v))
        if not edges:
            edges.append((0, 1))

        layout, cutwidth = solve_cmp_gurobi(
            n=n_nodes, edges=edges,
            time_limit=self.time_limit, threads=self.threads,
        )
        instance_dict = {
            "name": f"synthetic_{n_nodes}",
            "num_nodes": n_nodes,
            "num_edges": len(edges),
            "edges": [[int(u), int(v)] for u, v in edges],
            "solution": [int(i) for i in layout],
            "objective": float(cutwidth),
        }
        return instance_dict, layout, float(cutwidth)
