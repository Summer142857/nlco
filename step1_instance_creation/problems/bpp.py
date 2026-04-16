"""
BPP (Bin Packing Problem) instance extractor.
"""
import json
import random
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from pathlib import Path

from ..utils.base import InstanceExtractor
from ..utils.io import get_random_dataset_file
from ..solvers.solvers import solve_bin_packing_gurobi, solve_bin_packing_first_fit_decreasing


class BPPDatasetParser:
    """Parser for BPP dataset format."""

    def __init__(self, dataset_path: str):
        self.original_path = dataset_path
        # dataset_path should be a specific file at this point
        self.dataset_path = Path(dataset_path)
        self.instances = self._load_instances()

    def _load_instances(self) -> List[Dict[str, Any]]:
        """Load and parse all instances from the dataset file."""
        instances = []

        with open(self.dataset_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        i = 0
        # First line contains the number of instances in the file
        if i >= len(lines):
            return instances

        try:
            num_instances_in_file = int(lines[i])
            i += 1
        except ValueError:
            print(f"Warning: Cannot parse number of instances from first line: {lines[0]}")
            return instances

        # Parse each instance
        for instance_idx in range(num_instances_in_file):
            try:
                if i >= len(lines):
                    print(f"Warning: Unexpected end of file while parsing instance {instance_idx + 1}")
                    break

                # Read instance name
                instance_name = lines[i].strip()
                i += 1

                if i >= len(lines):
                    print(f"Warning: Missing instance info for {instance_name}")
                    break

                # Read instance info: bin_capacity num_items optimal_bins
                instance_info = lines[i].strip().split()
                if len(instance_info) < 3:
                    print(f"Warning: Invalid instance info format for {instance_name}: {lines[i]}")
                    i += 1
                    continue

                bin_capacity = float(instance_info[0])
                n_items = int(instance_info[1])
                optimal_bins = int(instance_info[2])
                i += 1

                # Read item sizes
                item_sizes = []
                for j in range(n_items):
                    if i >= len(lines):
                        print(f"Warning: Not enough item sizes for {instance_name}, expected {n_items}, got {len(item_sizes)}")
                        break

                    try:
                        item_size = float(lines[i])
                        item_sizes.append(item_size)
                        i += 1
                    except ValueError:
                        print(f"Warning: Cannot parse item size '{lines[i]}' for instance {instance_name}")
                        i += 1
                        break

                # Only add if we successfully parsed all items
                if len(item_sizes) == n_items:
                    instances.append({
                        'name': instance_name,
                        'bin_capacity': bin_capacity,
                        'item_sizes': item_sizes,
                        'n_items': n_items,
                        'optimal_bins': optimal_bins
                    })
                else:
                    print(f"Warning: Incomplete instance {instance_name}, expected {n_items} items, got {len(item_sizes)}")

            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing instance {instance_idx + 1} starting at line {i}: {e}")
                # Try to skip to next instance by looking for next instance name pattern
                i += 1
                continue

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


class BPPInstanceExtractor(InstanceExtractor):
    """BPP-specific instance extractor."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.is_directory = Path(dataset_path).is_dir()

        # If it's a single file, load once. If directory, we'll load per-instance
        if not self.is_directory:
            self.parser = BPPDatasetParser(dataset_path)
        else:
            self.parser = None

    def extract_instance(self, n_items: int) -> Tuple[List[float], List[List[int]], int, float]:
        """Extract a BPP instance with n_items.

        Args:
            n_items: Number of items to sample from the instance

        Returns:
            Tuple of (item_sizes, optimal_packing, num_bins_used, bin_capacity)
        """
        # If directory, randomly select a file for each instance extraction
        if self.is_directory:
            random_file = get_random_dataset_file(self.dataset_path)
            parser = BPPDatasetParser(random_file)
        else:
            parser = self.parser

        # Get a random instance from the dataset
        instance_data = parser.get_random_instance()
        original_items = instance_data['item_sizes']
        bin_capacity = instance_data['bin_capacity']

        # Ensure we don't sample more items than available
        if n_items > len(original_items):
            raise ValueError(f"Requested {n_items} items but instance only has {len(original_items)}")

        # Randomly sample n_items from the instance
        indices = random.sample(range(len(original_items)), n_items)
        sampled_items = [original_items[i] for i in indices]

        # Solve the sampled instance using Gurobi
        try:
            optimal_packing, num_bins = solve_bin_packing_gurobi(sampled_items, bin_capacity)
        except Exception as e:
            print(f"Gurobi solver failed: {e}, falling back to heuristic")
            optimal_packing, num_bins = solve_bin_packing_first_fit_decreasing(sampled_items, bin_capacity)

        return sampled_items, optimal_packing, num_bins, bin_capacity

    def get_problem_type(self) -> str:
        return "BPP"


class InstanceGenerator:
    """Main class for generating BPP problem instances."""

    def __init__(self, extractor: InstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances: int = 1) -> List[Dict[str, Any]]:
        """Generate multiple instances.

        Args:
            n_nodes: Number of items per instance (int) or tuple (min_items, max_items) for range
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

                if self.extractor.get_problem_type() == "BPP":
                    item_sizes, optimal_packing, num_bins, bin_capacity = result
                    instance_dict = {
                        'instance': item_sizes,  # Already a list
                        'solution': optimal_packing,
                        'obj': num_bins,
                        'bin_capacity': bin_capacity,
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

    def generate_and_save(self, n_nodes, n_instances: int, output_path: str = None,
                          out_path: str = None, **kwargs):
        """Generate instances and save them to JSON."""
        # Support both output_path and out_path keyword arguments
        path = out_path if out_path is not None else output_path
        if path is None:
            raise ValueError("Must provide output_path or out_path")
        if isinstance(n_nodes, tuple):
            print(f"Generating {n_instances} instances with {n_nodes[0]}-{n_nodes[1]} items each...")
        else:
            print(f"Generating {n_instances} instances with {n_nodes} items each...")
        instances = self.generate_instances(n_nodes, n_instances)
        self.save_instances_to_json(instances, path)
        return instances


def create_bpp_generator(dataset_path: str) -> InstanceGenerator:
    """Convenience function to create a BPP instance generator."""
    extractor = BPPInstanceExtractor(dataset_path)
    return InstanceGenerator(extractor)
