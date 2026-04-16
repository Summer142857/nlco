"""
KP (0-1 Knapsack Problem) instance generator.
"""
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..utils.io import parse_range


def sample_from(spec: Union[int, Tuple[int, int]], rng: np.random.Generator) -> int:
    """Sample an integer from a fixed value or a [lo,hi] inclusive range."""
    if isinstance(spec, int):
        return spec
    lo, hi = spec
    return int(rng.integers(lo, hi + 1))


class KnapsackGenerator:
    """Generate 0-1 Knapsack instances (uniform weights/values) and solve optimally."""

    def __init__(self, R: int = 1000, seed: Optional[int] = None):
        self.R = int(R)
        self.rng = np.random.default_rng(seed)

    def generate_instance(
        self,
        n_items: int,
        alpha: float,
        time_limit: float = 60.0,
        threads: int = 8,
    ) -> Dict:
        R = self.R
        n_items = int(n_items)

        # weights & profits ~ UniformInt{1..R}
        w = self.rng.integers(1, R + 1, size=n_items, dtype=int)
        p = self.rng.integers(1, R + 1, size=n_items, dtype=int)

        # capacity
        C = max(1, int(round(alpha * np.sum(w))))

        sol, obj, t = self.solve_optimal(w, p, C, time_limit=time_limit, threads=threads)

        return {
            "problem_type": "KP",
            "weights": w.tolist(),
            "profits": p.tolist(),
            "capacity": C,
            "solution": sol,
            "objective": obj,
            "solve_time": t,
            "n_items": n_items,
            "R": R,
            "alpha": float(alpha),
        }

    def generate_instances(
        self,
        n_items: Union[int, Tuple[int, int]],
        n_instances: int = 1,
        alpha_values: Optional[List[float]] = None,
        time_limit: float = 60.0,
        threads: int = 8,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """Generate multiple KP instances in the same format as generate_instance."""
        rng = random.Random(seed if seed is not None else 42)
        candidates = [float(a) for a in (alpha_values or [0.5])]

        instances = []
        for _ in range(int(n_instances)):
            if isinstance(n_items, tuple):
                current_n = rng.randint(n_items[0], n_items[1])
            else:
                current_n = int(n_items)

            alpha = rng.choice(candidates)
            inst = self.generate_instance(
                n_items=current_n,
                alpha=alpha,
                time_limit=time_limit,
                threads=threads,
            )
            instances.append(inst)

        return instances

    def generate_and_save(
        self,
        n_items: Union[int, Tuple[int, int]],
        n_instances: int,
        out_path: str,
        alpha_values: Optional[List[float]] = None,
        time_limit: float = 60.0,
        threads: int = 8,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """Generate multiple KP instances and save them to JSON."""
        instances = self.generate_instances(
            n_items=n_items,
            n_instances=n_instances,
            alpha_values=alpha_values,
            time_limit=time_limit,
            threads=threads,
            seed=seed,
        )
        out_path_p = Path(out_path)
        out_path_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path_p, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"[Info] Saved {len(instances)} KP instances to {out_path_p}")
        return instances

    @staticmethod
    def solve_optimal(
        weights: np.ndarray,
        profits: np.ndarray,
        capacity: int,
        time_limit: float = 60.0,
        threads: int = 8,
    ) -> Tuple[List[int], int, float]:
        n = int(len(weights))
        t0 = time.time()

        model = gp.Model("KP")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = float(time_limit)
        model.Params.Threads = int(threads)

        x = model.addVars(n, vtype=GRB.BINARY, name="x")

        model.addConstr(gp.quicksum(float(weights[i]) * x[i] for i in range(n)) <= float(capacity), name="capacity")
        model.setObjective(gp.quicksum(float(profits[i]) * x[i] for i in range(n)), GRB.MAXIMIZE)

        model.optimize()
        solve_time = time.time() - t0

        if model.SolCount > 0:
            sol_bits = [int(x[i].X > 0.5) for i in range(n)]
            obj = int(round(model.ObjVal))
        else:
            sol_bits, obj = [0] * n, 0

        return sol_bits, obj, solve_time
