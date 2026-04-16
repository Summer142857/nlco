"""
2SP (2D Strip Packing Problem) instance extractor.
"""
import math
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from pathlib import Path

from ..utils.base import InstanceExtractor
from ..utils.io import get_random_dataset_file
from ..solvers.solvers import solve_2sp_gurobi


def compute_area(items):
    return sum(it['width'] * it['height'] * it['demand'] for it in items)


def scaled_max_height(original_items, sampled_items, W, H_original, min_gap_ratio=1.0):
    """
    Scale max height while ensuring non-trivial lower bounds.
    """
    A_full = compute_area(original_items)
    A_sub  = compute_area(sampled_items)

    LB_full = max(1, math.ceil(A_full / W))
    LB_sub  = max(1, math.ceil(A_sub / W))

    rho = H_original / LB_full  # how tight the original was

    H_sub = math.ceil(rho * LB_sub)
    tallest = max(it['height'] for it in sampled_items)

    H_sub = max(H_sub, math.ceil(tallest * min_gap_ratio))

    return H_sub


class TwoSPDatasetParser:
    """Parser for 2D Strip Packing (2SP) dataset format."""

    def __init__(self, dataset_path: str):
        self.original_path = dataset_path
        self.dataset_path = Path(dataset_path)
        self.instances = self._load_instances()

    def _load_instances(self) -> List[Dict[str, Any]]:
        """Load and parse 2SP instance from the dataset file.

        Format:
        - Line 1: m (number of items)
        - Line 2: W H (width and height of bin, H=-1 for strip packing)
        - Next m lines: i w_i h_i d_i b_i p_i
        """
        instances = []

        with open(self.dataset_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 3:
            print(f"Warning: Not enough lines in {self.dataset_path.name}")
            return instances

        try:
            n_items = int(lines[0])

            parts = lines[1].split()
            bin_width = int(parts[0])
            bin_height = int(parts[1])

            items = []
            for i in range(2, min(2 + n_items, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 6:
                    items.append({
                        'id': int(parts[0]),
                        'width': int(parts[1]),
                        'height': int(parts[2]),
                        'demand': int(parts[3]),
                        'max_copies': int(parts[4]),
                        'profit': int(parts[5])
                    })

            if len(items) == n_items:
                instances.append({
                    'name': self.dataset_path.stem,
                    'bin_width': bin_width,
                    'bin_height': bin_height,
                    'n_items': n_items,
                    'items': items,
                    'is_strip_packing': (bin_height == -1)
                })
            else:
                print(f"Warning: Expected {n_items} items but got {len(items)} in {self.dataset_path.name}")

        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing 2SP file {self.dataset_path.name}: {e}")

        return instances

    def get_random_instance(self) -> Dict[str, Any]:
        """Get a random instance from the dataset."""
        if not self.instances:
            raise ValueError(f"No valid instances found in {self.dataset_path}")
        return random.choice(self.instances)


class TwoSPInstanceExtractor(InstanceExtractor):
    """2SP (2D Strip Packing) specific instance extractor."""

    def __init__(self, dataset_path: str, time_limit: float = 120.0, threads: int = 8):
        self.dataset_path = dataset_path
        self.is_directory = Path(dataset_path).is_dir()
        self.time_limit = time_limit
        self.threads = threads

        self.min_util = 0.60
        self.max_util = 0.95
        self.max_resample_tries = 50

        self.height_gap_ratio = 1.3

        self.min_tall_items = 2
        self.tall_item_ratio = 0.7
        self.stacked_height_multiplier = 1.5
        self.bin_width_scale = 0.8

        self.min_height_gap_ratio = 1.1
        self.max_quality_retries = 30

        if not self.is_directory:
            self.parser = TwoSPDatasetParser(dataset_path)
        else:
            self.parser = None

    def extract_instance(self, n_items: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, int, bool]:
        """Extract a 2SP instance with n_items item types."""
        if self.is_directory:
            random_file = get_random_dataset_file(self.dataset_path)
            parser = TwoSPDatasetParser(random_file)
        else:
            parser = self.parser

        instance_data = parser.get_random_instance()
        original_items = instance_data['items']
        original_bin_width = instance_data['bin_width']
        bin_height = instance_data['bin_height']
        is_strip_packing = instance_data['is_strip_packing']

        bin_width = int(original_bin_width * self.bin_width_scale)

        if n_items > len(original_items):
            raise ValueError(f"Requested {n_items} item types but instance only has {len(original_items)}")

        for quality_attempt in range(self.max_quality_retries):
            sampled_items = None
            max_height = None

            for _ in range(self.max_resample_tries):
                indices = random.sample(range(len(original_items)), n_items)
                candidate_items = [original_items[i] for i in indices]

                if is_strip_packing:
                    sampled_items = candidate_items
                    max_height = None
                    break

                candidate_max_h = scaled_max_height(
                    original_items=original_items,
                    sampled_items=candidate_items,
                    W=bin_width,
                    H_original=bin_height,
                    min_gap_ratio=self.height_gap_ratio
                )

                tallest = max(it['height'] for it in candidate_items)
                widest = max(it['width'] for it in candidate_items)
                if widest > bin_width:
                    continue
                if candidate_max_h < tallest:
                    continue

                heights = sorted([it['height'] for it in candidate_items], reverse=True)
                tallest_height = heights[0]
                tall_items_count = sum(1 for h in heights if h >= tallest_height * self.tall_item_ratio)

                if tall_items_count < self.min_tall_items:
                    continue

                total_stacked_height = sum(it['height'] * it['demand'] for it in candidate_items)
                if total_stacked_height < candidate_max_h * self.stacked_height_multiplier:
                    continue

                total_width_demand = sum(it['width'] * it['demand'] for it in candidate_items)
                min_rows_needed = math.ceil(total_width_demand / bin_width)

                if min_rows_needed < 2:
                    continue

                area = compute_area(candidate_items)
                util = area / (bin_width * candidate_max_h)

                if self.min_util <= util <= self.max_util:
                    sampled_items = candidate_items
                    max_height = candidate_max_h
                    break

            if sampled_items is None:
                indices = random.sample(range(len(original_items)), n_items)
                sampled_items = [original_items[i] for i in indices]
                if is_strip_packing:
                    max_height = None
                else:
                    max_height = scaled_max_height(
                        original_items=original_items,
                        sampled_items=sampled_items,
                        W=bin_width,
                        H_original=bin_height,
                        min_gap_ratio=self.height_gap_ratio
                    )

            solver_items = [(item['width'], item['height'], item['demand']) for item in sampled_items]

            try:
                packed_items, height_used = solve_2sp_gurobi(
                    solver_items,
                    bin_width,
                    max_height=max_height,
                    time_limit=self.time_limit,
                    threads=self.threads
                )
            except Exception as e:
                print(f"2SP Gurobi solver failed: {e}")
                raise

            is_nontrivial = True
            if not is_strip_packing:
                tallest_item = max(item['height'] for item in sampled_items)
                height_ratio = height_used / tallest_item

                if height_ratio >= self.min_height_gap_ratio:
                    is_nontrivial = True
                    break
                elif quality_attempt < self.max_quality_retries - 1:
                    print(f"  Trivial instance detected (height_ratio={height_ratio:.2f}), "
                          f"retrying... (attempt {quality_attempt + 1}/{self.max_quality_retries})")
                    continue
                else:
                    print(f"  Warning: Max retries reached. Accepting instance with height_ratio={height_ratio:.2f}")
                    is_nontrivial = False
            else:
                is_nontrivial = True
                break

        instance_dict = {
            'items': [
                {'width': item['width'], 'height': item['height'], 'demand': item['demand']}
                for item in sampled_items
            ],
            'bin_width': bin_width,
            'bin_height': -1 if is_strip_packing else int(max_height),
            'is_strip_packing': is_strip_packing
        }

        return instance_dict, packed_items, height_used, bin_width, is_nontrivial

    def get_problem_type(self) -> str:
        return "2SP"


class InstanceGenerator:
    """Generator for 2SP problem instances."""

    def __init__(self, extractor: TwoSPInstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances: int = 1) -> List[Dict[str, Any]]:
        instances = []
        while len(instances) < n_instances:
            try:
                if isinstance(n_nodes, tuple):
                    current_n = random.randint(n_nodes[0], n_nodes[1])
                else:
                    current_n = n_nodes
                instance_data, packed_items, height_used, bin_width, is_nontrivial = self.extractor.extract_instance(current_n)
                instances.append({
                    'instance': {
                        'items': instance_data['items'],
                        'bin_width': bin_width,
                        'bin_height': instance_data['bin_height'],
                        'is_strip_packing': instance_data['is_strip_packing'],
                    },
                    'solution': packed_items,
                    'obj': height_used,
                    'problem_type': '2SP'
                })
                print(f"Generated {len(instances)}/{n_instances} 2SP instances")
            except Exception as e:
                print(f"Error generating 2SP instance: {e}")
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
# Synthetic 2SP Extractor (no dataset files required)
# ============================================================

class Synthetic2SPInstanceExtractor(InstanceExtractor):
    """Generate random 2D Strip Packing instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "2SP"

    def extract_instance(self, n_items: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, int, bool]:
        strip_width = 100
        widths = np.random.randint(10, 40, size=n_items).tolist()
        heights = np.random.randint(10, 40, size=n_items).tolist()
        # Each item type has demand=1 for simplicity
        items = [(w, h, 1) for w, h in zip(widths, heights)]

        packed_items, height_used = solve_2sp_gurobi(
            items, strip_width, max_height=None,
            time_limit=self.time_limit, threads=self.threads,
        )
        instance_dict = {
            'items': [{'width': w, 'height': h, 'demand': 1} for w, h in zip(widths, heights)],
            'bin_width': strip_width,
            'bin_height': -1,
            'is_strip_packing': True,
        }
        return instance_dict, packed_items, float(height_used), strip_width, True
