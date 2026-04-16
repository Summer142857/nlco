"""
TSP (Travelling Salesman Problem) instance extractor.
"""
import json
import random
import ast
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from pathlib import Path

from ..utils.base import InstanceExtractor
from ..utils.io import get_random_dataset_file
from ..solvers.solvers import lkh


class TSPDatasetParser:
    """Parser for TSPLib dataset format."""

    def __init__(self, dataset_path: str):
        self.original_path = dataset_path
        # dataset_path should be a specific file at this point
        self.dataset_path = Path(dataset_path)
        self.instances = self._load_instances()

    def _load_instances(self) -> List[Dict[str, Any]]:
        """Load and parse all instances from the dataset file."""
        instances = []

        with open(self.dataset_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse the line as a Python list
                    data = ast.literal_eval(line)
                    instance_name = data[0]
                    optimal_value = float(data[1])

                    # Extract coordinates (x1, y1, x2, y2, ...)
                    coords_flat = [float(x) for x in data[2:]]

                    # Convert to (x, y) pairs
                    coordinates = []
                    for i in range(0, len(coords_flat), 2):
                        coordinates.append([coords_flat[i], coords_flat[i + 1]])

                    instances.append({
                        'name': instance_name,
                        'optimal_value': optimal_value,
                        'coordinates': np.array(coordinates)
                    })

        return instances

    def get_random_instance(self) -> Dict[str, Any]:
        """Get a random instance from the dataset."""
        return random.choice(self.instances)

    def get_instance_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific instance by name."""
        for instance in self.instances:
            if instance['name'] == name:
                return instance
        return None

    def list_instances(self) -> List[str]:
        """List all available instance names."""
        return [instance['name'] for instance in self.instances]


class TSPInstanceExtractor(InstanceExtractor):
    """TSP-specific instance extractor."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.is_directory = Path(dataset_path).is_dir()

        # If it's a single file, load once. If directory, we'll load per-instance
        if not self.is_directory:
            self.parser = TSPDatasetParser(dataset_path)
        else:
            self.parser = None

    def extract_instance(self, n_nodes: int) -> Tuple[np.ndarray, List[int], float]:
        """Extract a TSP instance with n_nodes.

        Args:
            n_nodes: Number of nodes to sample from the instance

        Returns:
            Tuple of (normalized_coordinates, optimal_tour, objective_value)
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

        # Solve the sampled instance using LKH
        optimal_tour, objective_value = lkh(torch.tensor(normalized_coords, dtype=torch.float32))

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
        return "TSP"


class InstanceGenerator:
    """Main class for generating TSP problem instances."""

    def __init__(self, extractor: InstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances: int = 1) -> List[Dict[str, Any]]:
        """Generate multiple instances.

        Args:
            n_nodes: Number of nodes per instance (int) or tuple (min_nodes, max_nodes) for range
            n_instances: Number of instances to generate

        Returns:
            List of instance dictionaries with 'instance' and 'solution' keys
        """
        instances = []

        for i in range(n_instances):
            try:
                # Handle range of nodes
                if isinstance(n_nodes, tuple):
                    current_n_nodes = random.randint(n_nodes[0], n_nodes[1])
                else:
                    current_n_nodes = n_nodes

                result = self.extractor.extract_instance(current_n_nodes)

                if self.extractor.get_problem_type() == "TSP":
                    coordinates, solution, objective_value = result
                    instance_dict = {
                        'instance': coordinates.tolist(),
                        'solution': solution,
                        'obj': float(objective_value),
                        'problem_type': self.extractor.get_problem_type()
                    }
                else:
                    raise ValueError(f"Unknown problem type: {self.extractor.get_problem_type()}")

                instances.append(instance_dict)

            except Exception as e:
                print(f"Error generating instance {i}: {e}")
                continue

        return instances

    def save_instances_to_json(self, instances: List[Dict[str, Any]], output_path: str):
        """Save instances to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(instances, f, indent=2)

        print(f"Saved {len(instances)} instances to {output_path}")

    def generate_and_save(self, n_nodes, n_instances: int, out_path: str = None, output_path: str = None, **kwargs):
        """Generate instances and save them to JSON."""
        path = out_path if out_path is not None else output_path
        if path is None:
            raise ValueError("Must provide out_path or output_path")
        if isinstance(n_nodes, tuple):
            print(f"Generating {n_instances} instances with {n_nodes[0]}-{n_nodes[1]} nodes each...")
        else:
            print(f"Generating {n_instances} instances with {n_nodes} nodes each...")
        instances = self.generate_instances(n_nodes, n_instances)
        self.save_instances_to_json(instances, path)
        return instances


def create_tsp_generator(dataset_path: str) -> InstanceGenerator:
    """Convenience function to create a TSP instance generator."""
    extractor = TSPInstanceExtractor(dataset_path)
    return InstanceGenerator(extractor)
