"""
MLP (Minimum Latency Problem) instance extractor.

Uses TSP coordinate datasets but optimizes total arrival time (sum of prefix tour lengths)
rather than total tour length.
"""
import random
from typing import List, Tuple, Dict, Any

import numpy as np
from pathlib import Path

from ..utils.base import InstanceExtractor
from ..utils.io import get_random_dataset_file
from .tsp import TSPDatasetParser
from ..solvers.solvers import solve_mlp_gurobi


class MLPInstanceExtractor(InstanceExtractor):
    """MLP (Minimum Latency Problem) instance extractor.

    Uses the same data sources as TSP (coordinates), but solves with
    a different objective: minimize sum of arrival times rather than
    total tour length.
    """

    def __init__(self, dataset_path: str, time_limit: float = 60.0, threads: int = 8):
        self.dataset_path = dataset_path
        self.is_directory = Path(dataset_path).is_dir()
        self.time_limit = time_limit
        self.threads = threads

        # If it's a single file, load once. If directory, we'll load per-instance
        if not self.is_directory:
            self.parser = TSPDatasetParser(dataset_path)
        else:
            self.parser = None

    def extract_instance(self, n_nodes: int) -> Tuple[np.ndarray, List[int], float]:
        """Extract an MLP instance with n_nodes.

        Args:
            n_nodes: Number of nodes to sample from the instance

        Returns:
            Tuple of (normalized_coordinates, optimal_tour, objective_value)
            where objective_value is the sum of arrival times (total latency)
        """
        # If directory, randomly select a file for each instance extraction
        if self.is_directory:
            random_file = get_random_dataset_file(self.dataset_path)
            parser = TSPDatasetParser(random_file)
        else:
            parser = self.parser

        # Get a random instance from the dataset
        instance_data = parser.get_random_instance()
        original_coords = instance_data['coordinates']

        # Ensure we don't sample more nodes than available
        if n_nodes > len(original_coords):
            raise ValueError(f"Requested {n_nodes} nodes but instance only has {len(original_coords)}")

        # Randomly sample n_nodes from the instance
        indices = random.sample(range(len(original_coords)), n_nodes)
        sampled_coords = original_coords[indices]

        # Normalize coordinates to [0, 100]
        normalized_coords = self._normalize_coordinates(sampled_coords)

        # Solve MLP using Gurobi (LKH cannot solve MLP as it has different objective)
        optimal_tour, objective_value = solve_mlp_gurobi(
            normalized_coords,
            depot=0,
            time_limit=self.time_limit,
            threads=self.threads
        )

        return normalized_coords, optimal_tour, objective_value

    def _normalize_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Normalize coordinates to [0, 100] range and round to integers."""
        min_vals = coordinates.min(axis=0)
        max_vals = coordinates.max(axis=0)

        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)

        # Scale to [0, 100] and round to integers
        normalized = (coordinates - min_vals) / range_vals * 100
        normalized = np.round(normalized).astype(int)

        return normalized

    def get_problem_type(self) -> str:
        return "MLP"


class InstanceGenerator:
    """Generator for MLP problem instances."""

    def __init__(self, extractor: MLPInstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances: int = 1) -> List[Dict[str, Any]]:
        instances = []
        while len(instances) < n_instances:
            try:
                if isinstance(n_nodes, tuple):
                    current_n = random.randint(n_nodes[0], n_nodes[1])
                else:
                    current_n = n_nodes
                coordinates, solution, objective_value = self.extractor.extract_instance(current_n)
                instances.append({
                    'instance': coordinates.tolist(),
                    'solution': solution,
                    'depot': 0,
                    'obj': float(objective_value),
                    'problem_type': 'MLP'
                })
                print(f"Generated {len(instances)}/{n_instances} MLP instances")
            except Exception as e:
                print(f"Error generating MLP instance: {e}")
        return instances

    def generate_and_save(self, n_nodes, n_instances: int, out_path: str = None, output_path: str = None, **kwargs):
        import json
        from pathlib import Path
        instances = self.generate_instances(n_nodes, n_instances)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(instances, f, indent=2)
        print(f"Saved {len(instances)} instances to {p}")
        return instances
