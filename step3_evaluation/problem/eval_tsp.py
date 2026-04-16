from typing import Any, Dict, List, Set, FrozenSet, Tuple

import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # permutation / tour


def tsp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse TSP instance from `instance_variant`.

    Expected `instance_variant` format (already labeled), e.g.:

      {
        "problem_type": "TSP",
        "num_nodes": 10,
        "nodes": [
          {"id": 0, "x": 64, "y": 1},
          {"id": 1, "x": 20, "y": 77},
          ...
        ],
        "depot": 0
      }

    We return:
      {
        "nodes": {id_0: (x0, y0), id_1: (x1, y1), ...},
        "depot": depot_id
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for TSP.")

    nodes_raw = inst_var.get("nodes", [])
    nodes: Dict[Any, Tuple[float, float]] = {}

    for n in nodes_raw:
        node_id = n["id"]
        x = float(n["x"])
        y = float(n["y"])
        nodes[node_id] = (x, y)

    depot = inst_var.get("depot", None)
    if depot is None:
        if nodes:
            depot = next(iter(nodes.keys()))

    return {
        "nodes": nodes,   # dict: label -> (x, y)
        "depot": depot,
    }



def tsp_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check TSP feasibility in label space + unified violation pattern.

    Patterns for TSP (per taxonomy Global_1):
      - FormatError: invalid type/length, unknown node labels, instance mismatch
      - Global1: tour/permutation violations (start/end depot, missing/duplicate nodes, depot inside)
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]
    depot = instance["depot"]
    node_set: Set[Any] = set(nodes.keys())

    # ---- Format checks ----
    if not isinstance(solution, list) or len(solution) < 2:
        return False, PATTERN_FORMAT, "Solution must be a non-empty list of node identifiers (tour)."

    if depot not in node_set:
        return False, PATTERN_FORMAT, f"Depot {depot!r} is not a valid node id in the instance."

    # ---- Global1: start/end depot ----
    if solution[0] != depot or solution[-1] != depot:
        return False, PATTERN_GLOBAL1, f"Tour must start and end at the depot {depot!r}."

    interior = solution[1:-1]

    # ---- FormatError: unknown labels ----
    unknown = set(interior) - node_set
    if unknown:
        return False, PATTERN_FORMAT, f"Solution contains unknown node labels: {unknown!r}"

    # ---- Global1: depot not allowed in interior ----
    if depot in interior:
        return False, PATTERN_GLOBAL1, f"Depot {depot!r} must not be visited in the interior of the tour."

    non_depot_nodes = node_set - {depot}
    interior_set = set(interior)

    # ---- Global1: must visit all non-depot nodes ----
    missing = non_depot_nodes - interior_set
    if missing:
        return False, PATTERN_GLOBAL1, f"The tour does not visit all non-depot nodes: missing {missing!r}"

    # ---- Global1: no repeats ----
    if len(interior) != len(interior_set):
        return False, PATTERN_GLOBAL1, "Some nodes are visited more than once in the interior of the tour."

    return True, PATTERN_OK, "Feasible TSP tour."

import math

def tsp_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    TSP objective: total Euclidean tour length.

    Assumes `solution` is a feasible tour (starts/ends at depot, visits
    every non-depot node exactly once). We sum sqrt((dx)^2 + (dy)^2)
    over all consecutive pairs in the tour.
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]

    if not isinstance(solution, list) or len(solution) < 2:
        return float("inf")

    total_dist = 0.0
    for u, v in zip(solution[:-1], solution[1:]):
        if u not in nodes or v not in nodes:
            continue
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        dx = x1 - x2
        dy = y1 - y2
        total_dist += math.hypot(dx, dy)

    return float(total_dist)
