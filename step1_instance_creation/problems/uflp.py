"""
UFLP (Uncapacitated Facility Location Problem) instance extractor.
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


class UFLPDatasetParser:
    """Parser for UFLP (Uncapacitated Facility Location Problem) files."""

    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.instance = self._load_instance()

    def _load_instance(self) -> Dict[str, Any]:
        with open(self.path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines[0].startswith("FILE:"):
            raise ValueError(f"Expected 'FILE:' on first line, got: {lines[0]}")
        name = lines[0].split("FILE:")[-1].strip()

        parts = lines[1].split()
        n_facilities = int(parts[0])
        n_customers = int(parts[1])

        opening_costs = []
        connection_costs = []

        for i in range(2, 2 + n_facilities):
            parts = lines[i].split()
            facility_id = int(parts[0])
            opening_cost = int(parts[1])
            costs_to_customers = [int(x) for x in parts[2:]]

            if len(costs_to_customers) != n_customers:
                raise ValueError(
                    f"Facility {facility_id} has {len(costs_to_customers)} connection costs, "
                    f"expected {n_customers}"
                )

            opening_costs.append(opening_cost)
            connection_costs.append(costs_to_customers)

        opening_costs = np.array(opening_costs, dtype=int)
        connection_costs = np.array(connection_costs, dtype=int)

        return {
            'name': name,
            'n_facilities': n_facilities,
            'n_customers': n_customers,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs
        }

    def get_instance(self):
        return self.instance


class UFLPInstanceExtractor(InstanceExtractor):
    """
    UFLP instance extractor from OR-Library datasets.
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
        return "UFLP"

    def extract_instance(self, n_customers: int, n_facilities: int = None
                         ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        if n_customers < 1:
            raise ValueError("UFLP requires at least 1 customer.")

        if n_facilities is None:
            alpha = random.uniform(2.0, 2.67)
            n_facilities = max(1, int(round(n_customers / alpha)))

        if n_facilities < 1:
            raise ValueError("UFLP requires at least 1 facility.")

        if self.is_dir:
            files = list(self.dataset_path.glob("*"))
            files = [f for f in files if f.is_file()]
            if not files:
                raise ValueError(f"No files found in {self.dataset_path}")
            file_path = str(random.choice(files))
            print(f"Selected: {Path(file_path).name}")
            parser = UFLPDatasetParser(file_path)
        else:
            parser = UFLPDatasetParser(self.dataset_path)

        base_instance = parser.get_instance()
        opening_costs_full = base_instance['opening_costs']
        connection_costs_full = base_instance['connection_costs']
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
        connection_costs = connection_costs_full[np.ix_(facility_idx, customer_idx)]

        open_facilities, assignments, obj_value = solve_uflp_gurobi(
            opening_costs=opening_costs.tolist(),
            connection_costs=connection_costs,
            time_limit=self.time_limit,
            threads=self.threads
        )

        instance_dict = {
            "opening_costs": opening_costs.tolist(),
            "connection_costs": connection_costs.tolist(),
            "objective": float(obj_value)
        }

        solution = {
            "open_facilities": open_facilities,
            "assignments": assignments
        }

        return instance_dict, solution, float(obj_value)


class InstanceGenerator:
    """Generic instance generator for UFLP."""

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
# Synthetic UFLP Extractor (no dataset files required)
# ============================================================

class SyntheticUFLPInstanceExtractor(InstanceExtractor):
    """Generate random UFLP instances synthetically."""

    def __init__(self, time_limit: float = 60.0, threads: int = 8):
        self.time_limit = time_limit
        self.threads = threads

    def get_problem_type(self) -> str:
        return "UFLP"

    def extract_instance(self, n_customers: int) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        alpha = random.uniform(2.0, 2.67)
        n_facilities = max(1, int(round(n_customers / alpha)))
        opening_costs = (np.random.rand(n_facilities) * 90 + 10).tolist()
        connection_costs = (np.random.rand(n_facilities, n_customers) * 50 + 1)

        open_facilities, assignments, obj_value = solve_uflp_gurobi(
            opening_costs=opening_costs,
            connection_costs=connection_costs,
            time_limit=self.time_limit,
            threads=self.threads,
        )
        instance_dict = {
            "opening_costs": opening_costs,
            "connection_costs": connection_costs.tolist(),
            "objective": float(obj_value),
        }
        solution = {"open_facilities": open_facilities, "assignments": assignments}
        return instance_dict, solution, float(obj_value)
