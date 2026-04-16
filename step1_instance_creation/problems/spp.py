"""
SPP (Set Partitioning Problem) instance generator.

Generates SPP instances using clustered set sampling and solves with Gurobi.
"""
import json
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import matplotlib.pyplot as plt

try:
    import gurobipy as gp
    from gurobipy import GRB

    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("[WARN] gurobipy not found; instances will be generated without solving.")


def visualize_incidence_matrix(inst_wrap: Dict[str, Any], save_path: str, title: str = ""):
    """
    Visualize the SPP instance as a binary incidence matrix.

    - Rows = elements (1..num_elements)
    - Columns = sets (1..num_sets)
    - Entry (i, j) = 1 if element i is contained in set j, else 0.
    """
    inst = inst_wrap["instance"]
    num_elements = inst["num_elements"]
    num_sets = inst["num_sets"]
    sets_ = inst["sets"]

    import numpy as np

    mat = np.zeros((num_elements, num_sets), dtype=int)
    for S in sets_:
        j_idx = S["id"] - 1
        for e in S["elements"]:
            i_idx = e - 1
            mat[i_idx, j_idx] = 1

    plt.figure(figsize=(max(4, num_sets * 0.15), max(4, num_elements * 0.15)))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.colorbar(label="incidence (0/1)")

    plt.xlabel("Sets")
    plt.ylabel("Elements")
    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_bag_stats(inst_wrap: Dict[str, Any], save_path: str, title: str = ""):
    """
    Scatter plot of set size vs cost, optionally highlighting the chosen sets
    in the solution (if available).
    """
    inst = inst_wrap["instance"]
    sol = inst_wrap.get("solution", None)

    sets_ = inst["sets"]
    all_sizes = [len(S["elements"]) for S in sets_]
    all_costs = [S["cost"] for S in sets_]

    chosen_ids = set()
    if sol:
        chosen_ids = {s["id"] for s in sol}

    chosen_sizes = []
    chosen_costs = []

    for S in sets_:
        if S["id"] in chosen_ids:
            chosen_sizes.append(len(S["elements"]))
            chosen_costs.append(S["cost"])

    plt.figure(figsize=(5, 4))

    plt.scatter(all_sizes, all_costs, alpha=0.5, label="all sets")

    if chosen_sizes:
        plt.scatter(
            chosen_sizes,
            chosen_costs,
            marker="x",
            s=60,
            label="chosen sets",
        )

    plt.xlabel("Set size (number of elements)")
    plt.ylabel("Set cost")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def parse_range(s: str) -> Union[int, Tuple[int, int]]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return int(a), int(b)
    return int(s)


class SPPInstanceGenerator:
    """
    SPP (Set Partitioning Problem) instance generator tailored for LLM reasoning.


    {
        "problem_type": "SPP",
        "num_elements": M,
        "num_sets": N,
        "density": nnz / (M * N),
        "sets": [
            { "id": j, "elements": [e1, e2, ...], "cost": c_j, "singleton": bool? },
            ...
        ],
        "generator_config": {...},
        "viz_matrix": "...",
        "viz_bag_stats": "..."
    }
    """

    def __init__(
        self,
        items_min: int = 8,
        items_max: int = 25,
        bag_factor_min: float = 2.0,
        bag_factor_max: float = 5.0,
        high_cost: int = 10**4,
        cluster_sigma_ratio: float = 0.1,
        remove_duplicate_bags: bool = True,
        rng_seed: int = 42,
    ):
        self.items_min = items_min
        self.items_max = items_max
        self.bag_factor_min = bag_factor_min
        self.bag_factor_max = bag_factor_max
        self.high_cost = high_cost
        self.cluster_sigma_ratio = cluster_sigma_ratio
        self.remove_duplicate_bags = remove_duplicate_bags
        self.rng = random.Random(rng_seed)

    def _sample_items_clustered(self, num_items: int, set_size: int) -> List[int]:
        if set_size <= 0:
            return []

        center = self.rng.randint(1, num_items)

        sigma = max(1.0, self.cluster_sigma_ratio * num_items)

        weights = []
        for i in range(1, num_items + 1):
            dist2 = (i - center) ** 2
            w = math.exp(-dist2 / (2.0 * sigma * sigma))
            weights.append(w)

        remaining_items = list(range(1, num_items + 1))
        remaining_weights = weights[:]
        chosen: List[int] = []

        for _ in range(min(set_size, num_items)):
            total_w = sum(remaining_weights)
            if total_w <= 0:
                candidate = self.rng.choice(remaining_items)
                idx = remaining_items.index(candidate)
            else:
                r = self.rng.random() * total_w
                acc = 0.0
                idx = 0
                for j, w in enumerate(remaining_weights):
                    acc += w
                    if acc >= r:
                        idx = j
                        break

            chosen_item = remaining_items.pop(idx)
            remaining_weights.pop(idx)
            chosen.append(chosen_item)

            if len(chosen) >= set_size:
                break

        chosen.sort()
        return chosen

    def _generate_single_instance(
        self,
        num_items=None,
        num_bags=None,
    ) -> Dict[str, Any]:
        if num_items is None:
            num_items = self.rng.randint(self.items_min, self.items_max)

        if num_bags is None:
            f_min = self.bag_factor_min
            f_max = self.bag_factor_max
            factor = self.rng.uniform(f_min, f_max)
            num_bags = max(1, int(round(factor * num_items)))

        sets_: List[Dict[str, Any]] = []
        seen_sets = set()

        for _ in range(num_bags):
            d = self.rng.uniform(0.1, 0.3)
            size = max(1, int(round(d * num_items)))

            items_in_bag = self._sample_items_clustered(num_items, size)
            if not items_in_bag:
                continue

            key = tuple(items_in_bag)
            if self.remove_duplicate_bags and key in seen_sets:
                continue
            if self.remove_duplicate_bags:
                seen_sets.add(key)

            base_cost = self.rng.randint(1, 100)
            cost = base_cost * len(items_in_bag)

            sets_.append(
                {
                    "id": len(sets_) + 1,
                    "elements": items_in_bag,
                    "cost": int(cost),
                }
            )

        for item in range(1, num_items + 1):
            sets_.append(
                {
                    "id": len(sets_) + 1,
                    "elements": [item],
                    "cost": int(self.high_cost),
                    "singleton": True,
                }
            )

        num_sets_final = len(sets_)
        nnz = sum(len(S["elements"]) for S in sets_)
        density = nnz / (num_items * num_sets_final)

        instance = {
            "problem_type": "SPP",
            "num_elements": num_items,
            "num_sets": num_sets_final,
            "density": float(density),
            "sets": sets_,
            "generator_config": {
                "items_min": self.items_min,
                "items_max": self.items_max,
                "bag_factor_min": self.bag_factor_min,
                "bag_factor_max": self.bag_factor_max,
                "high_cost": self.high_cost,
                "cluster_sigma_ratio": self.cluster_sigma_ratio,
            },
        }

        return instance

    def _solve_spp(self, inst: Dict[str, Any]):
        if not HAS_GUROBI:
            return [], 0.0

        num_elements = inst["num_elements"]
        sets_ = inst["sets"]
        num_sets = len(sets_)

        element_to_sets: Dict[int, List[int]] = {i: [] for i in range(1, num_elements + 1)}
        for j, S in enumerate(sets_):
            for e in S["elements"]:
                element_to_sets[e].append(j)

        model = gp.Model("spp_generator")
        model.Params.OutputFlag = 0

        x = model.addVars(num_sets, vtype=GRB.BINARY, name="x")
        costs = [S["cost"] for S in sets_]

        for i in range(1, num_elements + 1):
            indices = element_to_sets[i]
            if not indices:
                model.addConstr(0 == 1, name=f"infeasible_element_{i}")
            else:
                model.addConstr(
                    gp.quicksum(x[j] for j in indices) == 1,
                    name=f"partition_element_{i}",
                )

        model.setObjective(
            gp.quicksum(costs[j] * x[j] for j in range(num_sets)), GRB.MINIMIZE
        )
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print(f"[WARN] Gurobi SPP status {model.status}, returning empty solution.")
            return [], 0.0

        selected = [sets_[j]["id"] for j in range(num_sets) if x[j].X > 0.5]
        solution = [{"id": int(sid)} for sid in selected]
        obj_val = float(model.objVal)

        return solution, obj_val

    def generate_instances(
        self,
        items_range: Union[int, Tuple[int, int]],
        n_instances: int,
        solve: bool = True,
    ) -> List[Dict[str, Any]]:
        results = []

        for _ in range(n_instances):
            if isinstance(items_range, tuple):
                num_items = self.rng.randint(*items_range)
            else:
                num_items = items_range

            instance = self._generate_single_instance(num_items=num_items)
            if solve:
                sol, obj = self._solve_spp(instance)
            else:
                sol, obj = [], 0.0

            results.append(
                {
                    "instance": instance,
                    "solution": sol,
                    "obj": obj,
                    "problem_type": "SPP",
                }
            )

        return results

    def save_to_json(self, instances, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"[Info] Saved {len(instances)} instances to {out_path}")

    def generate_and_save(
        self,
        items_range: Union[int, Tuple[int, int]],
        n_instances: int,
        out_path: str,
        solve: bool = True,
        viz_dir=None,
    ):
        out_path = Path(out_path)
        data = self.generate_instances(items_range, n_instances, solve=solve)

        if viz_dir is not None:
            viz_path = Path(viz_dir)
            viz_path.mkdir(parents=True, exist_ok=True)

            for idx, inst_wrap in enumerate(data):
                mat_path = viz_path / f"instance_{idx:04d}_matrix.png"
                visualize_incidence_matrix(
                    inst_wrap,
                    save_path=str(mat_path),
                    title=f"SPP incidence matrix #{idx}",
                )

                stats_path_png = viz_path / f"instance_{idx:04d}_bag_stats.png"
                visualize_bag_stats(
                    inst_wrap,
                    save_path=str(stats_path_png),
                    title=f"SPP set size vs cost #{idx}",
                )

                inst = inst_wrap["instance"]
                inst["viz_matrix"] = str(mat_path)
                inst["viz_bag_stats"] = str(stats_path_png)

        elements_counts: List[int] = []
        set_counts: List[int] = []
        density_list: List[float] = []
        obj_list: List[float] = []

        for wrap in data:
            inst = wrap["instance"]
            elements_counts.append(inst["num_elements"])
            set_counts.append(inst["num_sets"])
            density_list.append(inst["density"])
            obj_list.append(wrap.get("obj", 0.0))

        stats: Dict[str, Any] = {
            "num_instances": len(data),
            "num_elements": {
                "avg": float(sum(elements_counts) / len(elements_counts)),
                "min": int(min(elements_counts)),
                "max": int(max(elements_counts)),
            },
            "num_sets": {
                "avg": float(sum(set_counts) / len(set_counts)),
                "min": int(min(set_counts)),
                "max": int(max(set_counts)),
            },
            "density": {
                "avg": float(sum(density_list) / len(density_list)),
                "min": float(min(density_list)),
                "max": float(max(density_list)),
            },
        }

        if HAS_GUROBI and obj_list:
            stats["obj"] = {
                "avg": float(sum(obj_list) / len(obj_list)),
                "min": float(min(obj_list)),
                "max": float(max(obj_list)),
            }

        self.save_to_json(data, out_path)
        print(stats)
        stats_path = out_path.with_name(out_path.stem + "_stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[Info] Saved stats to {stats_path}")

        return data
