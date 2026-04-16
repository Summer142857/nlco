from typing import Any, Dict, List, Set, Tuple
import pandas as pd
import math

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # tour / routing-permutation structure
PATTERN_GLOBAL2 = "Global2"  # capacity / budgeted subset


def cvrp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse CVRP instance from `instance_variant`.

    Expected `instance_variant` format (already labeled), e.g.:
      {
        "problem_type": "CVRP",
        "num_nodes": 10,
        "nodes": [
          {"id": 0, "x": 0, "y": 71, "demand": 0},
          {"id": 1, "x": 62, "y": 6,  "demand": 12},
          ...
        ],
        "depot": 0,
        "capacity": 100,
        "num_vehicles": 2,          # (recorded, NOT a constraint)
        "objective": 425.93,        # (recorded, not needed)
        "total_distance": 425.93    # (recorded, not needed)
      }

    Return:
      {
        "nodes": {id: (x, y), ...},
        "demands": {id: demand, ...},
        "depot": depot_id,
        "capacity": capacity
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for CVRP.")

    nodes_raw = inst_var.get("nodes", [])
    if not nodes_raw:
        raise ValueError("CVRP instance_variant has empty 'nodes'.")

    nodes: Dict[Any, Tuple[float, float]] = {}
    demands: Dict[Any, float] = {}

    for n in nodes_raw:
        node_id = n["id"]
        x = float(n["x"])
        y = float(n["y"])
        d = float(n.get("demand", 0.0))
        nodes[node_id] = (x, y)
        demands[node_id] = d

    depot = inst_var.get("depot", None)
    if depot is None:
        # fallback: choose first node id
        depot = nodes_raw[0]["id"]

    capacity = inst_var.get("capacity", None)
    if capacity is None:
        raise ValueError("CVRP instance_variant missing 'capacity'.")

    return {
        "nodes": nodes,        # dict: label -> (x, y)
        "demands": demands,    # dict: label -> demand
        "depot": depot,
        "capacity": float(capacity),
    }


def cvrp_check_feasibility(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check CVRP feasibility (vehicle count NOT fixed) + violation pattern.

    Patterns for CVRP (per taxonomy Global_{1,2}):
      - FormatError: invalid format/types/unknown node ids
      - Global1: routing/tour structure + each customer exactly once
      - Global2: capacity violation
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]
    demands: Dict[Any, float] = instance["demands"]
    depot = instance["depot"]
    capacity = float(instance["capacity"])
    node_set: Set[Any] = set(nodes.keys())

    # ---- Format checks ----
    if not isinstance(solution, list) or len(solution) == 0:
        return False, PATTERN_FORMAT, "Solution must be a non-empty list of routes."

    # If depot isn't even a known node, the instance/solution mismatch is not meaningfully checkable
    if depot not in node_set:
        return False, PATTERN_FORMAT, f"Depot {depot!r} is not a valid node id in the instance."

    all_customers = node_set - {depot}
    seen_customers: List[Any] = []

    for ridx, route in enumerate(solution):
        if not isinstance(route, list) or len(route) < 2:
            return False, PATTERN_FORMAT, f"Route {ridx} must be a list with length >= 2."

        # ---- Global1: route must be a depot-to-depot walk ----
        if route[0] != depot or route[-1] != depot:
            return False, PATTERN_GLOBAL1, f"Route {ridx} must start and end at depot {depot!r}."

        interior = route[1:-1]

        # unknown nodes in the interior => can't interpret against instance
        unknown = set(interior) - node_set
        if unknown:
            return False, PATTERN_FORMAT, f"Route {ridx} contains unknown node ids: {unknown!r}"

        # ---- Global1: depot cannot appear inside ----
        if depot in interior:
            return False, PATTERN_GLOBAL1, f"Depot {depot!r} must not appear in the interior of route {ridx}."

        # ---- Global2: capacity check ----
        load = 0.0
        for nid in interior:
            load += float(demands.get(nid, 0.0))
        if load > capacity + 1e-9:
            return (
                False,
                PATTERN_GLOBAL2,
                f"Route {ridx} exceeds capacity: load={load:.6f} > capacity={capacity:.6f}",
            )

        seen_customers.extend(interior)

    # ---- Global1: each customer exactly once across all routes ----
    seen_set = set(seen_customers)

    missing = all_customers - seen_set
    if missing:
        return False, PATTERN_GLOBAL1, f"Some customers are not visited: missing {missing!r}"

    if len(seen_customers) != len(seen_set):
        cnt: Dict[Any, int] = {}
        for x in seen_customers:
            cnt[x] = cnt.get(x, 0) + 1
        dup = {k for k, v in cnt.items() if v > 1}
        return False, PATTERN_GLOBAL1, f"Some customers are visited more than once: {dup!r}"

    return True, PATTERN_OK, "Feasible CVRP solution."


def cvrp_objective_model(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> float:
    """
    CVRP objective: minimize total Euclidean travel distance across all routes.

    Assumes solution is feasible. If not well-formed, return inf.
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]

    if not isinstance(solution, list) or len(solution) == 0:
        return float("inf")

    total = 0.0
    for route in solution:
        if not isinstance(route, list) or len(route) < 2:
            return float("inf")
        for u, v in zip(route[:-1], route[1:]):
            if u not in nodes or v not in nodes:
                # ill-formed route
                return float("inf")
            x1, y1 = nodes[u]
            x2, y2 = nodes[v]
            total += math.hypot(x1 - x2, y1 - y2)

    return float(total)
