"""
QAP (Quadratic Assignment Problem) instance extractor.
"""
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod

import gurobipy as gp
from gurobipy import GRB

from ..utils.base import InstanceExtractor
from ..utils.io import get_random_dataset_file


# ============================================================
# QAP Data Parser
# ============================================================

class QAPDatasetParser:
    """Parser for QAPLIB .dat format files."""

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instance = self._load_instance()

    def _load_instance(self) -> Dict[str, Any]:
        """
        Load a QAP instance from a QAPLIB .dat file.

        Expected format:
        - Line 1: dimension n
        - Line 2: blank (optional)
        - Next lines: distance matrix (n×n elements, possibly split across multiple lines)
        - Blank line (optional)
        - Next lines: flow matrix (n×n elements, possibly split across multiple lines)
        """
        with open(self.path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # First line: dimension
        n = int(lines[0])

        # Collect all remaining numbers
        all_numbers = []
        for line in lines[1:]:
            numbers = [int(x) for x in line.split()]
            all_numbers.extend(numbers)

        # We expect n*n numbers for distance matrix + n*n numbers for flow matrix
        expected_total = 2 * n * n
        if len(all_numbers) != expected_total:
            raise ValueError(
                f"Expected {expected_total} numbers (2 × {n} × {n}) for matrices, "
                f"but found {len(all_numbers)} numbers in file {self.path.name}"
            )

        # First n*n numbers: distance matrix
        dist_flat = all_numbers[:n * n]
        distance_matrix = np.array(dist_flat, dtype=int).reshape(n, n)

        # Next n*n numbers: flow matrix
        flow_flat = all_numbers[n * n:]
        flow_matrix = np.array(flow_flat, dtype=int).reshape(n, n)

        return {
            'name': self.path.stem,
            'size': n,
            'distance_matrix': distance_matrix,
            'flow_matrix': flow_matrix
        }

    def get_instance(self):
        return self.instance


# ============================================================
# QAP Solver
# ============================================================

def solve_qap_gurobi(
    distance_matrix: np.ndarray,
    flow_matrix: np.ndarray,
    time_limit: float = 60.0,
    threads: int = 8,
    mip_gap: float = 0.01
) -> Tuple[List[int], float]:
    """
    Solve QAP using Gurobi with Kaufman-Broeckx linearization.
    """
    n = distance_matrix.shape[0]

    # Create model
    model = gp.Model("QAP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)
    model.setParam('MIPGap', mip_gap)

    # Decision variables: x[i,k] = 1 if facility i is assigned to location k
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    # Linearization variables: y[i,j,k,l] = x[i,k] * x[j,l]
    # Only create variables for non-zero flow and distance entries to reduce model size
    y = {}
    for i in range(n):
        for j in range(n):
            if flow_matrix[i, j] != 0:
                for k in range(n):
                    for l in range(n):
                        if distance_matrix[k, l] != 0:
                            y[i, j, k, l] = model.addVar(
                                vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}_{l}"
                            )

    # Constraints: Each facility assigned to exactly one location
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(n)) == 1, name=f"facility_{i}")

    # Constraints: Each location has exactly one facility
    for k in range(n):
        model.addConstr(gp.quicksum(x[i, k] for i in range(n)) == 1, name=f"location_{k}")

    # Linearization constraints
    for i in range(n):
        for j in range(n):
            if flow_matrix[i, j] != 0:
                for k in range(n):
                    for l in range(n):
                        if distance_matrix[k, l] != 0:
                            model.addConstr(y[i, j, k, l] <= x[i, k],
                                            name=f"lin1_{i}_{j}_{k}_{l}")
                            model.addConstr(y[i, j, k, l] <= x[j, l],
                                            name=f"lin2_{i}_{j}_{k}_{l}")
                            model.addConstr(
                                y[i, j, k, l] >= x[i, k] + x[j, l] - 1,
                                name=f"lin3_{i}_{j}_{k}_{l}"
                            )

    # Objective: minimize total cost
    obj_expr = gp.quicksum(
        flow_matrix[i, j] * distance_matrix[k, l] * y[i, j, k, l]
        for i in range(n)
        for j in range(n)
        if flow_matrix[i, j] != 0
        for k in range(n)
        for l in range(n)
        if distance_matrix[k, l] != 0
    )
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract solution
        assignment = [-1] * n
        for i in range(n):
            for k in range(n):
                if x[i, k].X > 0.5:
                    assignment[i] = k
                    break

        # Verify assignment is valid
        if -1 in assignment:
            raise RuntimeError("Invalid assignment: some facilities not assigned")

        # Calculate objective value directly from the assignment
        obj_value = 0.0
        for i in range(n):
            for j in range(n):
                obj_value += flow_matrix[i, j] * distance_matrix[assignment[i], assignment[j]]

        return assignment, float(obj_value)
    else:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")


# ============================================================
# QAP Instance Extractor
# ============================================================

class QAPInstanceExtractor(InstanceExtractor):
    """
    QAP instance extractor from QAPLIB datasets.

    - Loads .dat files from QAPLIB format
    - Supports sub-sampling to create smaller instances
    - Uses Gurobi MILP (exact) solver
    """

    def __init__(
        self,
        dataset_path: str,
        time_limit: float = 60.0,
        threads: int = 8,
        mip_gap: float = 0.01
    ):
        self.dataset_path = Path(dataset_path)
        self.is_dir = self.dataset_path.is_dir()
        self.time_limit = time_limit
        self.threads = threads
        self.mip_gap = mip_gap

    def get_problem_type(self) -> str:
        return "QAP"

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        """
        Extract a QAP instance with n_nodes facilities/locations.
        """
        if n_nodes < 2:
            raise ValueError("QAP requires at least 2 facilities/locations.")

        # Load a random QAP instance file
        if self.is_dir:
            file_path = get_random_dataset_file(self.dataset_path)
            parser = QAPDatasetParser(file_path)
        else:
            parser = QAPDatasetParser(self.dataset_path)

        base_instance = parser.get_instance()
        dist_full = base_instance['distance_matrix']
        flow_full = base_instance['flow_matrix']
        n_full = base_instance['size']

        if n_nodes > n_full:
            raise ValueError(
                f"Requested {n_nodes} nodes, but instance {base_instance['name']} "
                f"only has {n_full} facilities/locations."
            )

        # Sub-sample if needed
        if n_nodes == n_full:
            idx = list(range(n_full))
        else:
            idx = sorted(random.sample(range(n_full), n_nodes))

        # Extract sub-matrices
        dist_matrix = dist_full[np.ix_(idx, idx)]
        flow_matrix = flow_full[np.ix_(idx, idx)]

        assignment, obj_value = solve_qap_gurobi(
            distance_matrix=dist_matrix,
            flow_matrix=flow_matrix,
            time_limit=self.time_limit,
            threads=self.threads,
            mip_gap=self.mip_gap
        )

        instance_dict = {
            "distance_matrix": dist_matrix.tolist(),
            "flow_matrix": flow_matrix.tolist(),
            "objective": float(obj_value)
        }

        return instance_dict, assignment, float(obj_value)


class InstanceGenerator:
    """Generate and save QAP instances (delegates to utils.base.InstanceGenerator)."""

    def __init__(self, extractor: QAPInstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances: int):
        from ..utils.base import InstanceGenerator as BaseGen
        gen = BaseGen(self.extractor)
        return gen.generate_instances(n_nodes, n_instances)

    def generate_and_save(self, n_nodes, n_instances: int, out_path: str = None,
                          output_path: str = None, oversample_factor: float = 3.0, **kwargs):
        from ..utils.base import InstanceGenerator as BaseGen
        path = out_path if out_path is not None else output_path
        if path is None:
            raise ValueError("Must provide out_path or output_path")
        gen = BaseGen(self.extractor)
        return gen.generate_and_save(n_nodes=n_nodes, n_instances=n_instances,
                                     out_path=path, oversample_factor=oversample_factor)


# ============================================================
# Synthetic QAP Extractor (no dataset files required)
# ============================================================

class SyntheticQAPInstanceExtractor:
    """Generate random QAP instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "QAP"

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        flow = np.random.randint(0, 100, size=(n_nodes, n_nodes))
        flow = (flow + flow.T) // 2  # symmetric
        np.fill_diagonal(flow, 0)
        dist = np.random.randint(0, 100, size=(n_nodes, n_nodes))
        dist = (dist + dist.T) // 2  # symmetric
        np.fill_diagonal(dist, 0)
        assignment, obj = solve_qap_gurobi(
            distance_matrix=dist,
            flow_matrix=flow,
            time_limit=self.time_limit,
            threads=self.threads,
        )
        instance_dict = {
            "flow_matrix": flow.tolist(),
            "distance_matrix": dist.tolist(),
            "n": n_nodes,
        }
        return instance_dict, assignment, float(obj)
