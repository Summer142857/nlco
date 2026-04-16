from typing import Any, Dict, List, Set, Tuple
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # permutation / tour
PATTERN_GLOBAL7 = "Global7"  # temporal consistency (time windows)


def tsptw_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse TSPTW instance from `instance_variant`.

    Expected instance_variant format, e.g.:
      {
        "problem_type": "TSPTW",
        "num_nodes": 6,
        "nodes": [
          {"id": 1, "x": 100, "y": 88, "tw_start": 0, "tw_end": 918},
          ...
        ],
        "depot": 1
      }

    Return:
      {
        "nodes": {id: (x, y), ...},
        "time_windows": {id: (tw_start, tw_end), ...},
        "depot": depot_id,
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for TSPTW.")

    nodes_raw = inst_var.get("nodes", [])
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("TSPTW instance_variant has no 'nodes' list.")

    nodes: Dict[Any, Tuple[float, float]] = {}
    time_windows: Dict[Any, Tuple[float, float]] = {}

    for n in nodes_raw:
        node_id = n["id"]
        x = float(n["x"])
        y = float(n["y"])
        tw_start = float(n["tw_start"])
        tw_end = float(n["tw_end"])
        nodes[node_id] = (x, y)
        time_windows[node_id] = (tw_start, tw_end)

    depot = inst_var.get("depot", None)
    if depot is None:
        depot = nodes_raw[0]["id"]

    if depot not in nodes:
        raise ValueError(f"Depot {depot!r} is not in nodes list.")

    return {
        "nodes": nodes,
        "time_windows": time_windows,
        "depot": depot,
    }



def tsptw_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check TSPTW feasibility + unified violation pattern.

    Patterns for TSPTW (per taxonomy Global_{1,7}):
      - FormatError: invalid format/types/unknown labels/instance mismatch
      - Global1: tour/permutation violations
      - Global7: time window violations (waiting allowed)
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]
    tws: Dict[Any, Tuple[float, float]] = instance["time_windows"]
    depot = instance["depot"]
    node_set: Set[Any] = set(nodes.keys())

    # ---- Format checks ----
    if not isinstance(solution, list) or len(solution) < 2:
        return False, PATTERN_FORMAT, "Solution must be a non-empty list of node identifiers (tour)."

    if depot not in node_set:
        return False, PATTERN_FORMAT, f"Depot {depot!r} is not a valid node id in the instance."

    # ---- Global1: start/end at depot ----
    if solution[0] != depot or solution[-1] != depot:
        return False, PATTERN_GLOBAL1, f"Tour must start and end at the depot {depot!r}."

    interior = solution[1:-1]

    # ---- FormatError: unknown labels anywhere in tour ----
    unknown_all = set(solution) - node_set
    if unknown_all:
        return False, PATTERN_FORMAT, f"Solution contains unknown node labels: {unknown_all!r}"

    # ---- Global1: depot must not appear in interior ----
    if depot in interior:
        return False, PATTERN_GLOBAL1, f"Depot {depot!r} must not be visited in the interior of the tour."

    # ---- Global1: visit each non-depot exactly once ----
    non_depot_nodes = node_set - {depot}
    interior_set = set(interior)

    missing = non_depot_nodes - interior_set
    if missing:
        return False, PATTERN_GLOBAL1, f"The tour does not visit all non-depot nodes: missing {missing!r}"

    if len(interior) != len(interior_set):
        return False, PATTERN_GLOBAL1, "Some nodes are visited more than once in the interior of the tour."

    # ---- Global7: time windows check (waiting allowed) ----
    # Guard: time window must exist for every visited node
    missing_tw = [v for v in solution if v not in tws]
    if missing_tw:
        return False, PATTERN_FORMAT, f"Instance/solution error: missing time window for nodes: {missing_tw!r}"

    time = 0.0

    # Enforce depot time window if present (it is, per guard above)
    dep_start, dep_end = tws[depot]
    if time < dep_start:
        time = dep_start
    if time > dep_end:
        return False, PATTERN_GLOBAL7, (
            f"Depot time window violated at start: arrival={time:.6f}, tw=[{dep_start},{dep_end}]"
        )

    try:
        for u, v in zip(solution[:-1], solution[1:]):
            x1, y1 = nodes[u]
            x2, y2 = nodes[v]
            dist = math.hypot(x1 - x2, y1 - y2)
            time += dist

            tw_start, tw_end = tws[v]
            if time < tw_start:
                time = tw_start  # wait
            if time > tw_end:
                return False, PATTERN_GLOBAL7, (
                    f"Time window violated at node {v!r}: arrival={time:.6f}, tw=[{tw_start},{tw_end}]"
                )
    except Exception as e:
        return False, PATTERN_FORMAT, f"Could not evaluate travel/time-windows due to invalid coordinates: {e!r}"

    return True, PATTERN_OK, "Feasible TSPTW tour."



def tsptw_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    Objective: minimize total Euclidean travel distance of the tour.
    (Waiting does not add distance.)
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]

    if not isinstance(solution, list) or len(solution) < 2:
        return float("inf")

    total_dist = 0.0
    for u, v in zip(solution[:-1], solution[1:]):
        if u not in nodes or v not in nodes:
            return float("inf")
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        total_dist += math.hypot(x1 - x2, y1 - y2)

    return float(total_dist)
