"""
GAP (Generalized Assignment Problem) instance extractor.
"""
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

from ..utils.base import InstanceExtractor
from ..solvers.solvers import solve_gap_gurobi


# ============================================================
# GAP Data Parser
# ============================================================

class GAPDatasetParser:
    """Parser for GAP (Generalized Assignment Problem) files."""

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instance = self._load_instance()

    def _load_instance(self) -> Dict[str, Any]:
        """
        Load a GAP instance from OR-Library format.

        Expected format:
        - Line 1: n_agents n_tasks
        - Next n_agents blocks: resource consumption values (n_tasks values per agent)
        - Next n_agents blocks: assignment costs (n_tasks values per agent)
        - Last line: capacity for each agent (n_agents values)
        """
        with open(self.path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # First line: n_agents n_tasks
        parts = lines[0].split()
        n_agents = int(parts[0])
        n_tasks = int(parts[1])

        # Collect all remaining numbers
        all_numbers = []
        for line in lines[1:]:
            numbers = [int(x) for x in line.split()]
            all_numbers.extend(numbers)

        # Expected structure:
        # - n_agents * n_tasks values for resource consumption
        # - n_agents * n_tasks values for assignment costs
        # - n_agents values for capacities
        expected_total = n_agents * n_tasks * 2 + n_agents

        if len(all_numbers) != expected_total:
            raise ValueError(
                f"Expected {expected_total} numbers ({n_agents * n_tasks * 2} + {n_agents}), "
                f"but found {len(all_numbers)} numbers in file {self.path.name}"
            )

        # First n_agents * n_tasks numbers: resource consumption
        resource_flat = all_numbers[:n_agents * n_tasks]
        resource_consumption = np.array(resource_flat, dtype=int).reshape(n_agents, n_tasks)

        # Next n_agents * n_tasks numbers: assignment costs
        cost_flat = all_numbers[n_agents * n_tasks:2 * n_agents * n_tasks]
        assignment_costs = np.array(cost_flat, dtype=int).reshape(n_agents, n_tasks)

        # Last n_agents numbers: capacities
        capacities = np.array(all_numbers[2 * n_agents * n_tasks:], dtype=int)

        return {
            'name': self.path.stem,
            'n_agents': n_agents,
            'n_tasks': n_tasks,
            'resource_consumption': resource_consumption,
            'assignment_costs': assignment_costs,
            'capacities': capacities
        }

    def get_instance(self):
        return self.instance


# ============================================================
# GAP Instance Extractor
# ============================================================

class GAPInstanceExtractor(InstanceExtractor):
    """
    GAP instance extractor from OR-Library datasets.

    - Loads GAP files from OR-Library format
    - Supports sub-sampling to create smaller instances
    - Solves using Gurobi MILP
    """

    def __init__(
        self,
        dataset_path: str,
        time_limit: float = 60.0,
        threads: int = 8
    ):
        self.dataset_path = Path(dataset_path)
        self.is_dir = self.dataset_path.is_dir()
        self.time_limit = time_limit
        self.threads = threads
        self._file_specs = None

    def _load_file_specs(self) -> List[Tuple[Path, int, int]]:
        if self._file_specs is not None:
            return self._file_specs

        if not self.is_dir:
            self._file_specs = []
            return self._file_specs

        specs: List[Tuple[Path, int, int]] = []
        for path in sorted(self.dataset_path.glob("*")):
            if not path.is_file():
                continue
            with open(path, "r") as f:
                first_line = f.readline().strip()
            if not first_line:
                continue
            parts = first_line.split()
            if len(parts) < 2:
                continue
            n_agents, n_tasks = int(parts[0]), int(parts[1])
            specs.append((path, n_agents, n_tasks))

        self._file_specs = specs
        return specs

    def get_problem_type(self) -> str:
        return "GAP"

    def extract_instance(self, n_tasks: int, n_agents: int = None) -> Tuple[Dict[str, Any], List[int], float]:
        """
        Extract a GAP instance with n_agents agents and n_tasks tasks.

        The number of agents is calculated as: n_agents = n_tasks / alpha
        where alpha is randomly sampled from [2.5, 3.5].
        """
        if n_tasks < 1:
            raise ValueError("GAP requires at least 1 task.")

        # Calculate n_agents from n_tasks / alpha if not provided
        if n_agents is None:
            alpha = random.uniform(2.5, 3.5)
            n_agents = max(2, int(round(n_tasks / alpha)))  # At least 2 agents

        if n_agents < 1:
            raise ValueError("GAP requires at least 1 agent.")

        # Load a random GAP instance file
        if self.is_dir:
            file_specs = self._load_file_specs()
            if not file_specs:
                raise ValueError(f"No files found in {self.dataset_path}")
            compatible_files = [
                path for path, max_agents, max_tasks in file_specs
                if max_agents >= n_agents and max_tasks >= n_tasks
            ]
            if not compatible_files:
                available_shapes = ", ".join(
                    f"{max_agents}x{max_tasks}"
                    for max_agents, max_tasks in sorted({
                        (max_agents, max_tasks) for _, max_agents, max_tasks in file_specs
                    })
                )
                raise ValueError(
                    f"No GAP dataset file in {self.dataset_path} can satisfy "
                    f"{n_agents} agents and {n_tasks} tasks. "
                    f"Available shapes: {available_shapes}."
                )
            file_path = str(random.choice(compatible_files))
            print(f"Selected: {Path(file_path).name}")
            parser = GAPDatasetParser(file_path)
        else:
            parser = GAPDatasetParser(self.dataset_path)

        base_instance = parser.get_instance()
        resource_full = base_instance['resource_consumption']
        costs_full = base_instance['assignment_costs']
        capacities_full = base_instance['capacities']
        n_agents_full = base_instance['n_agents']
        n_tasks_full = base_instance['n_tasks']

        if n_agents > n_agents_full:
            raise ValueError(
                f"Requested {n_agents} agents, but instance {base_instance['name']} "
                f"only has {n_agents_full} agents."
            )

        if n_tasks > n_tasks_full:
            raise ValueError(
                f"Requested {n_tasks} tasks, but instance {base_instance['name']} "
                f"only has {n_tasks_full} tasks."
            )

        # Sub-sample agents and tasks if needed
        if n_agents == n_agents_full:
            agent_idx = list(range(n_agents_full))
        else:
            agent_idx = sorted(random.sample(range(n_agents_full), n_agents))

        if n_tasks == n_tasks_full:
            task_idx = list(range(n_tasks_full))
        else:
            task_idx = sorted(random.sample(range(n_tasks_full), n_tasks))

        # Extract sub-instance
        resource_consumption = resource_full[np.ix_(agent_idx, task_idx)]
        assignment_costs = costs_full[np.ix_(agent_idx, task_idx)]
        capacities = capacities_full[agent_idx]

        # Solve GAP
        assignments, obj_value = solve_gap_gurobi(
            resource_consumption=resource_consumption,
            assignment_costs=assignment_costs,
            capacities=capacities.tolist(),
            time_limit=self.time_limit,
            threads=self.threads
        )

        instance_dict = {
            "resource_consumption": resource_consumption.tolist(),
            "assignment_costs": assignment_costs.tolist(),
            "capacities": capacities.tolist(),
            "objective": float(obj_value)
        }

        print(
            f"Generated GAP instance (n_agents={len(capacities)}, "
            f"n_tasks={n_tasks}, obj={obj_value:.2f})"
        )

        return instance_dict, assignments, float(obj_value)


# ============================================================
# Synthetic GAP Extractor (no dataset files required)
# ============================================================

class SyntheticGAPInstanceExtractor(InstanceExtractor):
    """Generate random GAP instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "GAP"

    def extract_instance(self, n_tasks: int, n_agents: int = None) -> Tuple[Dict[str, Any], List[int], float]:
        if n_agents is None:
            alpha = random.uniform(2.5, 3.5)
            n_agents = max(2, int(round(n_tasks / alpha)))
        resource_consumption = np.random.randint(1, 10, size=(n_agents, n_tasks))
        assignment_costs = np.random.randint(1, 50, size=(n_agents, n_tasks))
        # Capacity: ensure feasibility
        capacity_per_agent = max(resource_consumption.sum(axis=1).max(),
                                  int(resource_consumption.sum() / n_agents * 1.5))
        capacities = [int(capacity_per_agent)] * n_agents

        assignments, obj_value = solve_gap_gurobi(
            resource_consumption=resource_consumption,
            assignment_costs=assignment_costs,
            capacities=capacities,
            time_limit=self.time_limit,
            threads=self.threads,
        )
        instance_dict = {
            "resource_consumption": resource_consumption.tolist(),
            "assignment_costs": assignment_costs.tolist(),
            "capacities": capacities,
            "n_agents": n_agents,
            "n_tasks": n_tasks,
        }
        return instance_dict, assignments, float(obj_value)
