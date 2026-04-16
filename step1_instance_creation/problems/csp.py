"""
CSP (Cutting Stock Problem) instance extractor.
"""
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from pathlib import Path

from ..utils.base import InstanceExtractor
from ..utils.io import get_random_dataset_file
from ..solvers.solvers import solve_csp_gurobi, solve_bin_packing_gurobi, solve_bin_packing_first_fit_decreasing


class CSPDatasetParser:
    """Parser for CSP (Cutting Stock Problem) dataset format."""

    def __init__(self, dataset_path: str):
        self.original_path = dataset_path
        self.dataset_path = Path(dataset_path)
        self.instances = self._load_instances()

    def _load_instances(self) -> List[Dict[str, Any]]:
        """Load and parse CSP instance from the dataset file.

        Format:
        - Line 1: Number of item types
        - Line 2: Bin capacity
        - Subsequent lines: weight demand (one pair per line)
        """
        instances = []

        with open(self.dataset_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 3:
            print(f"Warning: Not enough lines in {self.dataset_path.name}")
            return instances

        try:
            # Line 1: number of item types
            n_types = int(lines[0])

            # Line 2: bin capacity
            bin_capacity = float(lines[1])

            # Remaining lines: weight-demand pairs
            weights = []
            demands = []

            for i in range(2, min(2 + n_types, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 2:
                    weight = int(parts[0])
                    demand = int(parts[1])
                    weights.append(weight)
                    demands.append(demand)

            if len(weights) == n_types:
                instances.append({
                    'bin_capacity': bin_capacity,
                    'weights': weights,
                    'demands': demands,
                    'n_types': n_types,
                    'total_items': sum(demands)
                })
            else:
                print(f"Warning: Expected {n_types} item types but got {len(weights)} in {self.dataset_path.name}")

        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing CSP file {self.dataset_path.name}: {e}")

        return instances

    def get_random_instance(self) -> Dict[str, Any]:
        """Get a random instance from the dataset."""
        if not self.instances:
            raise ValueError(f"No valid instances found in {self.dataset_path}")
        return random.choice(self.instances)


class CSPInstanceExtractor(InstanceExtractor):
    """CSP (Cutting Stock Problem) specific instance extractor."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.is_directory = Path(dataset_path).is_dir()

        # If it's a single file, load once. If directory, we'll load per-instance
        if not self.is_directory:
            self.parser = CSPDatasetParser(dataset_path)
        else:
            self.parser = None

    def extract_instance(self, n_types: int) -> Tuple[Dict[str, Any], List, int, float]:
        """Extract a CSP instance with n_types item types."""
        # If directory, randomly select a file for each instance extraction
        if self.is_directory:
            random_file = get_random_dataset_file(self.dataset_path)
            parser = CSPDatasetParser(random_file)
        else:
            parser = self.parser

        instance_data = parser.get_random_instance()
        original_weights = instance_data['weights']
        original_demands = instance_data['demands']
        bin_capacity = instance_data['bin_capacity']

        if n_types > len(original_weights):
            raise ValueError(f"Requested {n_types} item types but instance only has {len(original_weights)}")

        indices = random.sample(range(len(original_weights)), n_types)
        sampled_weights = [original_weights[i] for i in indices]
        sampled_demands = [original_demands[i] for i in indices]

        try:
            optimal_packing, num_bins = solve_csp_gurobi(
                sampled_weights,
                sampled_demands,
                int(bin_capacity)
            )
        except Exception as e:
            print(f"CSP Gurobi solver failed: {e}, falling back to BPP expansion")
            expanded_items = []
            type_map = []
            for type_idx, (weight, demand) in enumerate(zip(sampled_weights, sampled_demands)):
                for _ in range(demand):
                    expanded_items.append(weight)
                    type_map.append(type_idx)

            try:
                bpp_packing, num_bins = solve_bin_packing_gurobi(expanded_items, int(bin_capacity))
            except Exception as e2:
                print(f"BPP Gurobi also failed: {e2}, using FFD heuristic")
                bpp_packing, num_bins = solve_bin_packing_first_fit_decreasing(expanded_items, bin_capacity)

            optimal_packing = []
            for bin_items in bpp_packing:
                pattern_dict = {}
                for item_idx in bin_items:
                    type_idx = type_map[item_idx]
                    pattern_dict[type_idx] = pattern_dict.get(type_idx, 0) + 1
                optimal_packing.append(pattern_dict)

        csp_data = {
            'weights': sampled_weights,
            'demands': sampled_demands
        }

        return csp_data, optimal_packing, num_bins, bin_capacity

    def get_problem_type(self) -> str:
        return "CSP"


class InstanceGenerator:
    """Generator for CSP problem instances."""

    def __init__(self, extractor: CSPInstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances: int = 1) -> List[Dict[str, Any]]:
        instances = []
        while len(instances) < n_instances:
            try:
                if isinstance(n_nodes, tuple):
                    current_n = random.randint(n_nodes[0], n_nodes[1])
                else:
                    current_n = n_nodes
                csp_data, optimal_packing, num_bins, bin_capacity = self.extractor.extract_instance(current_n)
                instances.append({

                    'weights': csp_data['weights'],
                    'demands': csp_data['demands'],
                    'bin_capacity': float(bin_capacity),

                    'solution': optimal_packing,
                    'obj': num_bins,
                    'problem_type': 'CSP'
                })
                print(f"Generated {len(instances)}/{n_instances} CSP instances")
            except Exception as e:
                print(f"Error generating CSP instance: {e}")
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
# Synthetic CSP Extractor (no dataset files required)
# ============================================================

class SyntheticCSPInstanceExtractor(InstanceExtractor):
    """Generate random Cutting Stock Problem instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "CSP"

    def extract_instance(self, n_types: int) -> Tuple[Dict[str, Any], List, int, float]:
        bin_capacity = 100
        widths = np.random.randint(5, 40, size=n_types).tolist()
        demands = np.random.randint(1, 10, size=n_types).tolist()

        optimal_packing, num_bins = solve_csp_gurobi(widths, demands, bin_capacity)
        instance_dict = {
            "weights": widths,
            "demands": demands,
        }
        return instance_dict, optimal_packing, num_bins, float(bin_capacity)
