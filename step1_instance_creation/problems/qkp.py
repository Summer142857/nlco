"""
QKP (Quadratic Knapsack Problem) instance extractor.
"""
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from pathlib import Path

from ..utils.base import InstanceExtractor
from ..utils.io import get_random_dataset_file
from ..solvers.solvers import solve_qkp_gurobi


class QKPDatasetParser:
    """Parser for Quadratic Knapsack Problem (QKP) dataset format."""

    def __init__(self, dataset_path: str):
        self.original_path = dataset_path
        self.dataset_path = Path(dataset_path)
        self.instances = self._load_instances()

    def _load_instances(self) -> List[Dict[str, Any]]:
        """Load and parse QKP instance from the dataset file.

        Format (from QAPLIB):
        - Line 1: n (number of variables)
        - Line 2: linear coefficients c_i (space-separated, n values)
        - Lines 3 to n+2: quadratic coefficients (upper triangular, row by row)
        - Blank line
        - Next line: 0 (for <= constraint) or 1 (for = constraint)
        - Next line: capacity
        - Next line: weights a_i (space-separated, n values)
        """
        instances = []

        with open(self.dataset_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        if len(lines) < 3:
            print(f"Warning: Not enough lines in {self.dataset_path.name}")
            return instances

        try:
            n = int(lines[0])

            linear_coeffs = list(map(int, lines[1].split()))
            if len(linear_coeffs) != n:
                print(f"Warning: Expected {n} linear coefficients but got {len(linear_coeffs)}")
                return instances

            quadratic_coeffs = np.zeros((n, n), dtype=int)

            quad_lines = lines[2:2+n]
            all_quad_values = []
            for line in quad_lines:
                if line.strip():
                    values = list(map(int, line.split()))
                    all_quad_values.extend(values)

            idx = 0
            for i in range(n):
                for j in range(i, n):
                    if idx < len(all_quad_values):
                        quadratic_coeffs[i, j] = all_quad_values[idx]
                        idx += 1

            line_idx = 2 + n
            while line_idx < len(lines) and not lines[line_idx].strip():
                line_idx += 1

            if line_idx >= len(lines):
                print(f"Warning: Missing constraint type in {self.dataset_path.name}")
                return instances

            constraint_type = int(lines[line_idx])
            line_idx += 1

            if line_idx >= len(lines):
                print(f"Warning: Missing capacity in {self.dataset_path.name}")
                return instances
            capacity = int(lines[line_idx])
            line_idx += 1

            if line_idx >= len(lines):
                print(f"Warning: Missing weights in {self.dataset_path.name}")
                return instances
            weights = list(map(int, lines[line_idx].split()))
            if len(weights) != n:
                print(f"Warning: Expected {n} weights but got {len(weights)}")
                return instances

            instances.append({
                'name': self.dataset_path.stem,
                'n': n,
                'linear_coeffs': np.array(linear_coeffs, dtype=int),
                'quadratic_coeffs': quadratic_coeffs,
                'weights': np.array(weights, dtype=int),
                'capacity': capacity,
                'constraint_type': constraint_type
            })

        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing QKP file {self.dataset_path.name}: {e}")

        return instances

    def get_random_instance(self) -> Dict[str, Any]:
        """Get a random instance from the dataset."""
        if not self.instances:
            raise ValueError(f"No valid instances found in {self.dataset_path}")
        return random.choice(self.instances)


class QKPInstanceExtractor(InstanceExtractor):
    """QKP (Quadratic Knapsack Problem) specific instance extractor."""

    def __init__(self, dataset_path: str, time_limit: float = 60.0, threads: int = 8):
        self.dataset_path = dataset_path
        self.is_directory = Path(dataset_path).is_dir()
        self.time_limit = time_limit
        self.threads = threads

        if not self.is_directory:
            self.parser = QKPDatasetParser(dataset_path)
        else:
            self.parser = None

    def extract_instance(self, n_items: int) -> Tuple[Dict[str, Any], List[int], float]:
        """Extract a QKP instance with n_items variables."""
        if self.is_directory:
            random_file = get_random_dataset_file(self.dataset_path)
            parser = QKPDatasetParser(random_file)
        else:
            parser = self.parser

        instance_data = parser.get_random_instance()
        n_original = instance_data['n']

        if n_items > n_original:
            raise ValueError(f"Requested {n_items} items but instance only has {n_original}")

        indices = sorted(random.sample(range(n_original), n_items))

        sampled_linear = instance_data['linear_coeffs'][indices]
        sampled_quadratic = instance_data['quadratic_coeffs'][np.ix_(indices, indices)]
        sampled_weights = instance_data['weights'][indices]

        total_weight_original = instance_data['weights'].sum()
        total_weight_sampled = sampled_weights.sum()
        capacity_ratio = total_weight_sampled / total_weight_original
        capacity = int(instance_data['capacity'] * capacity_ratio)

        try:
            solution, objective = solve_qkp_gurobi(
                linear_coeffs=sampled_linear,
                quadratic_coeffs=sampled_quadratic,
                weights=sampled_weights,
                capacity=capacity,
                time_limit=self.time_limit,
                threads=self.threads
            )
            if len(solution) == 0 or len(solution) == 1:
                raise ValueError("invalid solution")
        except Exception as e:
            print(f"QKP Gurobi solver failed: {e}")
            raise

        instance_dict = {
            'linear_coeffs': sampled_linear.tolist(),
            'quadratic_coeffs': sampled_quadratic.tolist(),
            'weights': sampled_weights.tolist(),
            'capacity': int(capacity)
        }

        return instance_dict, solution, objective

    def get_problem_type(self) -> str:
        return "QKP"


class InstanceGenerator:
    """Generator for QKP problem instances."""

    def __init__(self, extractor: QKPInstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances: int = 1) -> List[Dict[str, Any]]:
        instances = []
        while len(instances) < n_instances:
            try:
                if isinstance(n_nodes, tuple):
                    current_n = random.randint(n_nodes[0], n_nodes[1])
                else:
                    current_n = n_nodes
                instance_data, solution, objective_value = self.extractor.extract_instance(current_n)
                instances.append({
                    'instance': {
                        'linear_coeffs': instance_data['linear_coeffs'],
                        'quadratic_coeffs': instance_data['quadratic_coeffs'],
                        'weights': instance_data['weights'],
                        'capacity': instance_data['capacity'],
                    },
                    'solution': solution,
                    'obj': objective_value,
                    'problem_type': 'QKP'
                })
                print(f"Generated {len(instances)}/{n_instances} QKP instances")
            except Exception as e:
                print(f"Error generating QKP instance: {e}")
        return instances

    def generate_and_save(self, n_nodes, n_instances: int, out_path: str = None, output_path: str = None, **kwargs):
        import json
        from pathlib import Path
        path = out_path if out_path is not None else output_path
        if path is None:
            raise ValueError("Must provide out_path or output_path")
        instances = self.generate_instances(n_nodes, n_instances)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(instances, f, indent=2)
        print(f"Saved {len(instances)} instances to {p}")
        return instances


# ============================================================
# Synthetic QKP Extractor (no dataset files required)
# ============================================================

class SyntheticQKPInstanceExtractor:
    """Generate random QKP instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "QKP"

    def extract_instance(self, n_items: int) -> Tuple[Dict[str, Any], List[int], float]:
        linear_coeffs = np.random.randint(1, 50, size=n_items)
        quadratic_coeffs = np.random.randint(0, 20, size=(n_items, n_items))
        quadratic_coeffs = (quadratic_coeffs + quadratic_coeffs.T) // 2
        np.fill_diagonal(quadratic_coeffs, 0)
        weights = np.random.randint(1, 20, size=n_items)
        capacity = int(weights.sum() * 0.5)

        solution, objective = solve_qkp_gurobi(
            linear_coeffs=linear_coeffs, quadratic_coeffs=quadratic_coeffs,
            weights=weights, capacity=capacity,
            time_limit=self.time_limit, threads=self.threads,
        )
        instance_dict = {
            "linear_coeffs": linear_coeffs.tolist(),
            "quadratic_coeffs": quadratic_coeffs.tolist(),
            "weights": weights.tolist(),
            "capacity": int(capacity),
        }
        return instance_dict, solution, float(objective)
