"""
LOP (Linear Ordering Problem) instance extractor.

Uses .mat-format LOP datasets from datasets/LOP/. Samples an n×n submatrix
and solves with Gurobi (MILP maximization).
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.io import get_random_dataset_file
from ..solvers.solvers import solve_linear_ordering_problem_gurobi


class LOPDatasetParser:
    """Parser for LOP dataset files (.mat format)."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.instances = self._load_instances()

    def _load_instances(self) -> List[Dict[str, Any]]:
        instances = []
        with open(self.dataset_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 2:
            print(f"Warning: Invalid LOP file format in {self.dataset_path}")
            return instances

        try:
            description = lines[0]
            n = int(lines[1])
            matrix_data = []
            for i in range(2, len(lines)):
                row_values = [float(x) for x in lines[i].split()]
                matrix_data.extend(row_values)

            expected_values = n * n
            if len(matrix_data) != expected_values:
                print(f"Warning: Expected {expected_values} matrix values for {n}x{n}, got {len(matrix_data)}")
                return instances

            cost_matrix = np.array(matrix_data).reshape(n, n)
            instance_name = self.dataset_path.stem
            instances.append({
                "name": instance_name,
                "description": description,
                "size": n,
                "cost_matrix": cost_matrix,
            })
        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing LOP file {self.dataset_path}: {e}")

        return instances

    def get_random_instance(self) -> Dict[str, Any]:
        return random.choice(self.instances)

    def list_instances(self) -> List[str]:
        return [inst["name"] for inst in self.instances]


class LOPInstanceExtractor:
    """LOP-specific instance extractor."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.is_directory = Path(dataset_path).is_dir()
        self.parser = None if self.is_directory else LOPDatasetParser(dataset_path)

    def extract_instance(self, n_nodes: int) -> Tuple[np.ndarray, List[int], float]:
        if self.is_directory:
            random_file = get_random_dataset_file(self.dataset_path)
            parser = LOPDatasetParser(random_file)
        else:
            parser = self.parser

        instance_data = parser.get_random_instance()
        original_matrix = instance_data["cost_matrix"]

        if n_nodes > original_matrix.shape[0]:
            raise ValueError(
                f"Requested {n_nodes} nodes but instance only has {original_matrix.shape[0]}"
            )

        indices = sorted(random.sample(range(original_matrix.shape[0]), n_nodes))
        submatrix = original_matrix[np.ix_(indices, indices)]

        optimal_ordering, objective_value = solve_linear_ordering_problem_gurobi(submatrix)
        return submatrix, optimal_ordering, float(objective_value)

    def get_problem_type(self) -> str:
        return "LOP"


class InstanceGenerator:
    """Generate and save LOP instances."""

    def __init__(self, extractor: LOPInstanceExtractor):
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
                cost_matrix, optimal_ordering, objective_value = self.extractor.extract_instance(current_n)
                instances.append({
                    "instance": cost_matrix.tolist(),
                    "solution": optimal_ordering,
                    "obj": objective_value,
                    "problem_type": "LOP",
                })
                if len(instances) % 10 == 0 or len(instances) == n_instances:
                    print(f"Generated {len(instances)}/{n_instances} LOP instances")
            except Exception as e:
                print(f"Error generating LOP instance: {e}")
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
            print(f"Generating {n_instances} LOP instances with {n_nodes[0]}-{n_nodes[1]} nodes...")
        else:
            print(f"Generating {n_instances} LOP instances with {n_nodes} nodes...")
        instances = self.generate_instances(n_nodes, n_instances)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"Saved {len(instances)} LOP instances to {out}")
        return instances
