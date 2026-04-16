"""
AP3 (3-Index Assignment Problem) instance extractor.
"""
import json
import random
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

import gurobipy as gp
from gurobipy import GRB

from ..utils.base import InstanceExtractor


# ============================================================
# AP3 Data Parser
# ============================================================

class AP3DatasetParser:
    """Parser for 3-index assignment (AP3) Balas-style files.

    Expected format:
    - File contains exactly n^3 integer cost values (no header).
    - We infer n as the integer cube root of the number of values.
    """

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instance = self._load_instance()

    def _load_instance(self) -> Dict[str, Any]:
        with open(self.path, "r") as f:
            numbers = [int(x) for x in f.read().split()]

        total = len(numbers)
        # Infer n from n^3 = total
        n_float = round(total ** (1.0 / 3.0))
        n = int(n_float)
        if n ** 3 != total:
            raise ValueError(
                f"AP3 file {self.path.name}: expected n^3 values but got {total}; "
                f"inferred n={n} gives {n**3}."
            )

        cost_tensor = np.array(numbers, dtype=int).reshape(n, n, n)

        return {
            "name": self.path.stem,
            "n": n,
            "cost_tensor": cost_tensor,
        }

    def get_instance(self) -> Dict[str, Any]:
        return self.instance


# ============================================================
# AP3 Solver
# ============================================================

def solve_ap3_gurobi(
    cost_tensor: np.ndarray,
    time_limit: float = 60.0,
    threads: int = 8,
) -> Tuple[List[Tuple[int, int, int]], float]:
    """
    Solve the 3-index assignment problem (AP3) using a direct MILP model.

    cost_tensor[i, j, k] is the cost of choosing triple (i, j, k).
    Constraints:
        - For each i: sum_{j, k} x[i, j, k] = 1
        - For each j: sum_{i, k} x[i, j, k] = 1
        - For each k: sum_{i, j} x[i, j, k] = 1
    """
    if cost_tensor.ndim != 3:
        raise ValueError("AP3 cost tensor must be 3-dimensional.")

    n1, n2, n3 = cost_tensor.shape
    if not (n1 == n2 == n3):
        raise ValueError(
            f"AP3 cost tensor must be cubic (n x n x n), got {cost_tensor.shape}."
        )
    n = n1

    model = gp.Model("AP3")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", threads)

    # Decision variables: x[i, j, k] = 1 if triple (i, j, k) is selected
    x = model.addVars(n, n, n, vtype=GRB.BINARY, name="x")

    # Each i must appear exactly once
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j, k] for j in range(n) for k in range(n)) == 1,
            name=f"assign_i_{i}",
        )

    # Each j must appear exactly once
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j, k] for i in range(n) for k in range(n)) == 1,
            name=f"assign_j_{j}",
        )

    # Each k must appear exactly once
    for k in range(n):
        model.addConstr(
            gp.quicksum(x[i, j, k] for i in range(n) for j in range(n)) == 1,
            name=f"assign_k_{k}",
        )

    # Objective: minimize sum c[i, j, k] * x[i, j, k]
    obj_expr = gp.quicksum(
        cost_tensor[i, j, k] * x[i, j, k]
        for i in range(n)
        for j in range(n)
        for k in range(n)
    )
    model.setObjective(obj_expr, GRB.MINIMIZE)

    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi optimization for AP3 failed with status {model.status}")

    # Extract chosen triples (i, j, k) with x[i, j, k] = 1
    assignment_triples: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if x[i, j, k].X > 0.5:
                    assignment_triples.append((i, j, k))

    # Basic sanity check: we expect exactly n triples
    if len(assignment_triples) != n:
        raise RuntimeError(
            f"Invalid AP3 solution: expected {n} triples, got {len(assignment_triples)}."
        )

    obj_value = float(model.ObjVal)
    return assignment_triples, obj_value


# ============================================================
# AP3 Instance Extractor
# ============================================================

class AP3InstanceExtractor(InstanceExtractor):
    """
    AP3 instance extractor from Balas-style datasets.

    - Loads files containing n^3 integer cost entries
    - Supports sub-sampling to create smaller n x n x n instances
    - Solves using Gurobi MILP
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
        self._file_inventory = None

    def get_problem_type(self) -> str:
        return "AP3"

    def _infer_file_n(self, path: Path) -> int:
        """Infer AP3 dimension from the filename when possible, else parse the file."""
        match = re.search(r"-n(\d+)-", path.name)
        if match:
            return int(match.group(1))
        return int(AP3DatasetParser(str(path)).get_instance()["n"])

    def _list_dataset_files(self) -> List[Tuple[Path, int]]:
        if self._file_inventory is None:
            files = [f for f in self.dataset_path.glob("*") if f.is_file()]
            self._file_inventory = [(f, self._infer_file_n(f)) for f in files]
        return self._file_inventory

    def _select_random_file(self, min_n: int) -> str:
        """Select a random AP3 file whose native dimension is at least min_n."""
        files = self._list_dataset_files()
        if not files:
            raise ValueError(f"No files found in {self.dataset_path}")
        eligible = [path for path, n in files if n >= min_n]
        if not eligible:
            available = sorted({n for _, n in files})
            raise ValueError(
                f"No AP3 dataset file in {self.dataset_path} can satisfy requested n={min_n}. "
                f"Available base sizes: {available}."
            )
        selected = random.choice(eligible)
        print(f"Selected AP3 file: {selected.name}")
        return str(selected)

    def extract_instance(self, n: int) -> Tuple[Dict[str, Any], List[Tuple[int, int, int]], float]:
        """
        Extract an AP3 instance of dimension n (n x n x n).

        If the base instance has dimension N >= n, we sub-sample n indices
        from {0, ..., N-1} and restrict the cost tensor to that subset.
        """
        if n < 2:
            raise ValueError("AP3 requires at least dimension n >= 2.")

        # Load a random AP3 instance file or the provided file
        if self.is_dir:
            file_path = self._select_random_file(min_n=n)
            parser = AP3DatasetParser(file_path)
        else:
            parser = AP3DatasetParser(str(self.dataset_path))

        base_instance = parser.get_instance()
        cost_full = base_instance["cost_tensor"]
        n_full = base_instance["n"]

        if n > n_full:
            raise ValueError(
                f"Requested n={n} but instance {base_instance['name']} only has n={n_full}."
            )

        # Sub-sample indices if needed
        if n == n_full:
            idx = list(range(n_full))
        else:
            idx = sorted(random.sample(range(n_full), n))

        # Restrict cost tensor to selected indices along all three dimensions
        cost_tensor = cost_full[np.ix_(idx, idx, idx)]

        # Solve AP3 on this sub-instance
        assignment, obj_value = solve_ap3_gurobi(
            cost_tensor=cost_tensor,
            time_limit=self.time_limit,
            threads=self.threads,
        )

        instance_dict = {
            "cost_tensor": cost_tensor.tolist(),
            "objective": float(obj_value),
        }

        print(f"Generated AP3 instance (n={n}, obj={obj_value:.2f})")

        return instance_dict, assignment, float(obj_value)


# ============================================================
# Synthetic AP3 Extractor (no dataset files required)
# ============================================================

class SyntheticAP3InstanceExtractor(InstanceExtractor):
    """Generate random 3-Dimensional Assignment Problem instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "AP3"

    def extract_instance(self, n: int) -> Tuple[Dict[str, Any], List[Tuple[int, int, int]], float]:
        cost_tensor = np.random.randint(1, 100, size=(n, n, n))
        assignment, obj = solve_ap3_gurobi(
            cost_tensor=cost_tensor,
            time_limit=self.time_limit,
            threads=self.threads,
        )
        instance_dict = {"cost_tensor": cost_tensor.tolist(), "objective": float(obj)}
        return instance_dict, assignment, float(obj)
