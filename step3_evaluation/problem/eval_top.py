from typing import Any, Dict, List, Set, Tuple
import math
import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field


def top_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse TOP instance from `instance_variant`.

    Expected instance_variant format:
      {
        "problem_type": "TOP",
        "num_nodes": 27,
        "nodes": [{"id": 0, "x": 1, "y": 89, "prize": 0}, ...],
        "depot": 0,
        "max_length_per_vehicle": 225.0,
        "n_vehicles": 2
      }

    Returns:
      {
        "nodes": {id: (x, y)},
        "prizes": {id: prize},
        "depot": depot_id,
        "max_length_per_vehicle": float,
        "n_vehicles": int
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for TOP.")

    nodes_raw = inst_var.get("nodes", [])
    nodes: Dict[Any, Tuple[float, float]] = {}
    prizes: Dict[Any, float] = {}

    for n in nodes_raw:
        node_id = n["id"]
        nodes[node_id] = (float(n["x"]), float(n["y"]))
        prizes[node_id] = float(n.get("prize", 0.0))

    depot = inst_var.get("depot", None)
    if depot is None:
        if nodes:
            depot = next(iter(nodes.keys()))
        else:
            raise ValueError("TOP instance has no nodes and no depot.")

    max_len = float(inst_var.get("max_length_per_vehicle", float("inf")))
    n_vehicles = int(inst_var.get("n_vehicles", 1))

    return {
        "nodes": nodes,
        "prizes": prizes,
        "depot": depot,
        "max_length_per_vehicle": max_len,
        "n_vehicles": n_vehicles,
    }


def _euclid_route_length(route: List[Any], nodes: Dict[Any, Tuple[float, float]]) -> float:
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        if u not in nodes or v not in nodes:
            return float("inf")
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        total += math.hypot(x1 - x2, y1 - y2)
    return float(total)


def top_check_feasibility(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Check TOP feasibility in label space.

    Expected solution format:
      solution = [
        [depot, v1, v2, ..., depot],
        [depot, u1, u2, ..., depot],
        ...
      ]

    Constraints:
      1) solution is a list of routes; number of routes == n_vehicles
      2) each route is a list length >= 2 and starts/ends at depot
      3) route interior nodes:
         - all known labels
         - depot not in interior
         - no duplicates within a route
      4) no customer visited by more than one vehicle (global no-duplicate), excluding depot
      5) each route length <= max_length_per_vehicle
    """
    nodes = instance["nodes"]
    depot = instance["depot"]
    max_len = float(instance["max_length_per_vehicle"])
    n_vehicles = int(instance["n_vehicles"])
    node_set: Set[Any] = set(nodes.keys())

    if not isinstance(solution, list):
        return False, "Solution must be a list of routes (one per vehicle)."

    if len(solution) != n_vehicles:
        return False, f"Solution must contain exactly {n_vehicles} routes (got {len(solution)})."

    global_visited: Set[Any] = set()

    for ridx, route in enumerate(solution):
        if not isinstance(route, list) or len(route) < 2:
            return False, f"Route {ridx} must be a list with at least [depot, depot]."

        if route[0] != depot or route[-1] != depot:
            return False, f"Route {ridx} must start and end at depot {depot!r}."

        interior = route[1:-1]

        # unknown labels
        unknown = set(interior) - node_set
        if unknown:
            return False, f"Route {ridx} contains unknown node labels: {unknown!r}"

        # depot not allowed inside
        if depot in interior:
            return False, f"Depot {depot!r} must not appear in the interior of route {ridx}."

        # no duplicates inside route
        if len(interior) != len(set(interior)):
            return False, f"Route {ridx} visits some nodes more than once."

        # global no-duplicate (excluding depot)
        overlap = set(interior) & global_visited
        if overlap:
            return False, f"Some nodes are visited by multiple routes: {overlap!r}"

        global_visited |= set(interior)

        # length constraint
        rlen = _euclid_route_length(route, nodes)
        if rlen > max_len + 1e-9:
            return False, (
                f"Route {ridx} length {rlen:.6f} exceeds max_length_per_vehicle {max_len:.6f}."
            )

    return True, "Feasible TOP solution."


def top_objective_model(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> float:
    """
    TOP objective (to MAXIMIZE): total collected prize across all visited customers (unique),
    excluding depot.

    Assumes feasibility: no customer appears in more than one route.
    """
    prizes = instance["prizes"]
    depot = instance["depot"]

    if not isinstance(solution, list):
        return float("-inf")

    visited: Set[Any] = set()
    for route in solution:
        if not isinstance(route, list) or len(route) < 2:
            continue
        for v in route[1:-1]:
            if v != depot:
                visited.add(v)

    total_prize = 0.0
    for v in visited:
        total_prize += float(prizes.get(v, 0.0))

    return float(total_prize)
