"""
CFLP (Capacitated Facility Location Problem) instance extractor.
"""
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod

from ..solvers.solvers import (
    solve_uflp_gurobi,
    solve_cflp_gurobi,
    solve_pmed_gurobi,
    solve_mdp_gurobi,
    solve_pcenter_gurobi,
)


class InstanceExtractor(ABC):
    """Abstract base class for problem instance extractors."""

    @abstractmethod
    def extract_instance(self, n_nodes: int):
        """Return (instance_dict, solution, objective_value)."""
        pass

    @abstractmethod
    def get_problem_type(self) -> str:
        pass


class CFLPDatasetParser:
    """Parser for CFLP (Capacitated Facility Location Problem) files."""

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instance = self._load_instance()

    def _load_instance(self) -> Dict[str, Any]:
        with open(self.path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        parts = lines[0].split()
        n_facilities = int(parts[0])
        n_customers = int(parts[1])

        opening_costs = []
        capacities = []

        for i in range(1, 1 + n_facilities):
            parts = lines[i].split()
            opening_cost = float(parts[0])
            capacity = float(parts[1])
            opening_costs.append(opening_cost)
            capacities.append(capacity)

        all_remaining_values = []
        for i in range(1 + n_facilities, len(lines)):
            parts = lines[i].split()
            for val in parts:
                all_remaining_values.append(float(val))

        if len(all_remaining_values) < n_customers:
            raise ValueError(
                f"Not enough values for customer demands. Expected at least {n_customers}, "
                f"found {len(all_remaining_values)} in file {self.path.name}"
            )

        demands = all_remaining_values[:n_customers]

        connection_cost_values = all_remaining_values[n_customers:]
        expected_connection_costs = n_facilities * n_customers

        if len(connection_cost_values) != expected_connection_costs:
            raise ValueError(
                f"Expected {expected_connection_costs} connection costs ({n_facilities} x {n_customers}), "
                f"but found {len(connection_cost_values)} values in file {self.path.name}"
            )

        connection_costs = np.array(connection_cost_values).reshape(n_facilities, n_customers)
        demands = np.array(demands)

        return {
            'name': self.path.stem,
            'n_facilities': n_facilities,
            'n_customers': n_customers,
            'opening_costs': np.array(opening_costs),
            'capacities': np.array(capacities),
            'connection_costs': connection_costs,
            'demands': np.array(demands)
        }

    def get_instance(self):
        return self.instance


class CFLPInstanceExtractor(InstanceExtractor):
    """
    CFLP instance extractor from standard CFLP datasets.
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

    def get_problem_type(self) -> str:
        return "CFLP"

    def extract_instance(self, n_customers: int) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        if n_customers < 1:
            raise ValueError("CFLP requires at least 1 customer.")

        alpha = random.uniform(2.0, 2.67)
        n_facilities = max(1, int(round(n_customers / alpha)))

        if self.is_dir:
            files = list(self.dataset_path.glob("*"))
            files = [f for f in files if f.is_file()]
            if not files:
                raise ValueError(f"No files found in {self.dataset_path}")
            file_path = str(random.choice(files))
            print(f"Selected: {Path(file_path).name}")
            parser = CFLPDatasetParser(file_path)
        else:
            parser = CFLPDatasetParser(self.dataset_path)

        base_instance = parser.get_instance()
        opening_costs_full = base_instance['opening_costs']
        capacities_full = base_instance['capacities']
        connection_costs_full = base_instance['connection_costs']
        demands_full = base_instance['demands']
        n_facilities_full = base_instance['n_facilities']
        n_customers_full = base_instance['n_customers']

        if n_facilities > n_facilities_full:
            raise ValueError(
                f"Requested {n_facilities} facilities, but instance {base_instance['name']} "
                f"only has {n_facilities_full} facilities."
            )

        if n_customers > n_customers_full:
            raise ValueError(
                f"Requested {n_customers} customers, but instance {base_instance['name']} "
                f"only has {n_customers_full} customers."
            )

        if n_facilities == n_facilities_full:
            facility_idx = list(range(n_facilities_full))
        else:
            facility_idx = sorted(random.sample(range(n_facilities_full), n_facilities))

        if n_customers == n_customers_full:
            customer_idx = list(range(n_customers_full))
        else:
            customer_idx = sorted(random.sample(range(n_customers_full), n_customers))

        opening_costs = opening_costs_full[facility_idx]
        capacities = capacities_full[facility_idx]
        connection_costs = connection_costs_full[np.ix_(facility_idx, customer_idx)]
        demands = demands_full[customer_idx]

        open_facilities, assignments, obj_value = solve_cflp_gurobi(
            opening_costs=opening_costs.tolist(),
            capacities=capacities.tolist(),
            connection_costs=connection_costs,
            demands=demands.tolist(),
            time_limit=self.time_limit,
            threads=self.threads
        )

        instance_dict = {
            "opening_costs": opening_costs.tolist(),
            "capacities": capacities.tolist(),
            "connection_costs": connection_costs.tolist(),
            "demands": demands.tolist(),
            "objective": float(obj_value)
        }

        solution = {
            "open_facilities": open_facilities,
            "assignments": assignments
        }

        return instance_dict, solution, float(obj_value)


class InstanceGenerator:
    """Generic instance generator for CFLP."""

    def __init__(self, extractor: InstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances):
        results = []
        attempts = 0
        max_attempts = n_instances * 10

        while len(results) < n_instances and attempts < max_attempts:
            attempts += 1
            if isinstance(n_nodes, tuple):
                n = random.randint(*n_nodes)
            else:
                n = n_nodes

            try:
                ptype = self.extractor.get_problem_type()
                if ptype in ["UFLP", "CFLP"]:
                    inst, sol, obj = self.extractor.extract_instance(n_customers=n)
                    n_facilities = len(inst['opening_costs'])
                    alpha = n / n_facilities if n_facilities > 0 else 0
                    size_str = f"n_customers={n}, n_facilities={n_facilities}, alpha={alpha:.2f}"
                elif ptype in ["PMED", "PCENTER"]:
                    inst, sol, obj = self.extractor.extract_instance(n_vertices=n)
                    p = inst['p']
                    size_str = f"n_vertices={n}, p={p}"
                elif ptype == "MDP":
                    inst, sol, obj = self.extractor.extract_instance(n_vertices=n)
                    m = inst['m']
                    size_str = f"n_vertices={n}, m={m}"
                else:
                    inst, sol, obj = self.extractor.extract_instance(n)
                    size_str = f"n={n}"

                results.append(
                    {
                        "instance": inst,
                        "solution": sol,
                        "obj": obj,
                        "problem_type": ptype,
                    }
                )
                print(f"Generated instance {len(results)}/{n_instances} ({size_str}, obj={obj:.2f})")
            except Exception as e:
                print(f"Error generating instance (attempt {attempts}): {e}")
                continue

        if len(results) < n_instances:
            print(
                f"Warning: requested {n_instances} instances but only "
                f"generated {len(results)} after {attempts} attempts."
            )

        return results

    def save_to_json(self, instances, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"Saved {len(instances)} instances to {out_path}")

    def generate_and_save(self, n_nodes, n_instances, out_path, **kwargs):
        data = self.generate_instances(n_nodes, n_instances)
        self.save_to_json(data, out_path)
        return data


# ============================================================
# Synthetic CFLP Extractor (no dataset files required)
# ============================================================

class SyntheticCFLPInstanceExtractor(InstanceExtractor):
    """Generate random CFLP instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "CFLP"

    def extract_instance(self, n_customers: int) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        alpha = random.uniform(2.0, 2.67)
        n_facilities = max(1, int(round(n_customers / alpha)))
        opening_costs = (np.random.rand(n_facilities) * 90 + 10).tolist()
        demands = (np.random.randint(5, 20, size=n_customers)).tolist()
        total_demand = sum(demands)
        cap_per_fac = max(1, int(total_demand / n_facilities * 1.5))
        capacities = [cap_per_fac] * n_facilities
        connection_costs = (np.random.rand(n_facilities, n_customers) * 50 + 1)

        open_facilities, assignments, obj_value = solve_cflp_gurobi(
            opening_costs=opening_costs,
            capacities=capacities,
            connection_costs=connection_costs,
            demands=demands,
            time_limit=self.time_limit,
            threads=self.threads,
        )
        instance_dict = {
            "opening_costs": opening_costs,
            "capacities": capacities,
            "connection_costs": connection_costs.tolist(),
            "demands": demands,
            "objective": float(obj_value),
        }
        solution = {"open_facilities": open_facilities, "assignments": assignments}
        return instance_dict, solution, float(obj_value)
