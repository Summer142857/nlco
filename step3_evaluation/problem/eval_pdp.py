from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # permutation / tour
PATTERN_GLOBAL7 = "Global7"  # temporal consistency (precedence + time windows)


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)


def pdp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse PDP instance from `instance_variant`.

    Expected instance_variant format (label space):
      {
        "problem_type": "PDP",
        "num_nodes": 21,
        "depot": 1,
        "nodes": [
          {"id": 1, "x": 58, "y": 36, "tw_start": 0, "tw_end": 1236},
          ...
        ],
        "pickup_delivery_pairs": [[3,2], [20,21], ...],
      }

    Returns a compact instance dict for eval:
      - depot: node id
      - coords: dict[id] -> (x,y)
      - time_windows: dict[id] -> (start,end)
      - pairs: list[(pickup, delivery)]
      - all_nodes: set of all ids (including depot)
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for PDP.")

    depot = inst_var.get("depot", None)
    nodes_raw = inst_var.get("nodes", None)
    pairs_raw = inst_var.get("pickup_delivery_pairs", None)

    if depot is None:
        raise ValueError("PDP instance_variant missing 'depot'.")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("PDP instance_variant missing/invalid 'nodes'.")
    if not isinstance(pairs_raw, list):
        raise ValueError("PDP instance_variant missing/invalid 'pickup_delivery_pairs'.")

    coords: Dict[Any, Tuple[float, float]] = {}
    time_windows: Dict[Any, Tuple[float, float]] = {}
    all_nodes: Set[Any] = set()

    for n in nodes_raw:
        nid = n["id"]
        x = float(n["x"])
        y = float(n["y"])
        tw_start = float(n["tw_start"])
        tw_end = float(n["tw_end"])
        coords[nid] = (x, y)
        time_windows[nid] = (tw_start, tw_end)
        all_nodes.add(nid)

    # Normalize pairs to list of tuples
    pairs: List[Tuple[Any, Any]] = []
    for p in pairs_raw:
        if not (isinstance(p, list) or isinstance(p, tuple)) or len(p) != 2:
            raise ValueError(f"Invalid pickup_delivery_pairs entry: {p!r}")
        pairs.append((p[0], p[1]))

    if depot not in all_nodes:
        raise ValueError(f"Depot {depot!r} not found in nodes list.")

    return {
        "depot": depot,
        "coords": coords,
        "time_windows": time_windows,
        "pairs": pairs,
        "all_nodes": all_nodes,
    }



def pdp_check_feasibility(
    solution: Any,
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check feasibility for single-vehicle PDP with Time Windows + Precedence,
    with unified violation patterns.

    Patterns for PDP (taxonomy Global_{1,7,2}; this checker covers Global1 + Global7):
      - FormatError: wrong shape/types, nested routes, unknown nodes
      - Global1: route/tour violations (start/end depot, missing/duplicate nodes, depot inside)
      - Global7: precedence/time-window violations
    """
    # ---------- shape check ----------
    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a single route list [depot, ..., depot]."

    if not solution:
        return False, PATTERN_FORMAT, "Route is empty."

    # Reject nested routes explicitly
    if any(isinstance(x, list) for x in solution):
        return False, PATTERN_FORMAT, "Nested routes ([[...]]) are not supported; expected a flat list."

    route: List[Any] = solution

    depot = instance["depot"]
    coords = instance["coords"]
    tw = instance["time_windows"]
    pairs = instance["pairs"]
    all_nodes: Set[Any] = instance["all_nodes"]

    # ---------- Global1: start / end ----------
    if route[0] != depot or route[-1] != depot:
        return False, PATTERN_GLOBAL1, f"Route must start and end at depot {depot!r}."

    # ---------- FormatError: unknown nodes ----------
    unknown = set(route) - all_nodes
    if unknown:
        return False, PATTERN_FORMAT, f"Route contains unknown node IDs: {unknown!r}"

    # ---------- Global1: must visit each non-depot node exactly once ----------
    inner = route[1:-1]
    if depot in inner:
        return False, PATTERN_GLOBAL1, "Depot appears in the middle of the route."

    non_depot = all_nodes - {depot}

    # duplicates
    counts: Dict[Any, int] = {}
    for v in inner:
        counts[v] = counts.get(v, 0) + 1
    dup = {v: c for v, c in counts.items() if c > 1}
    if dup:
        return False, PATTERN_GLOBAL1, f"Some nodes are visited more than once: {dup!r}"

    # missing
    missing = non_depot - set(inner)
    if missing:
        return False, PATTERN_GLOBAL1, f"Some required nodes are missing: {missing!r}"

    # ---------- Global7: precedence (pickup before delivery) ----------
    pos = {v: i for i, v in enumerate(route)}
    for p, d in pairs:
        if pos[p] > pos[d]:
            return False, PATTERN_GLOBAL7, f"Pickup {p!r} occurs after delivery {d!r}."

    # ---------- Global7: time windows (waiting allowed) ----------
    t = 0.0
    for i in range(len(route)):
        v = route[i]
        if i == 0:
            arrival = 0.0
        else:
            prev = route[i - 1]
            arrival = t + _euclid(coords[prev], coords[v])

        start, end = tw[v]
        if arrival > end + 1e-9:
            return False, PATTERN_GLOBAL7, (
                f"Time window violated at node {v!r}: arrival {arrival:.3f} > latest {end:.3f}"
            )
        t = max(arrival, start)  # waiting allowed

    return True, PATTERN_OK, "Feasible PDP route."



def pdp_objective_model(
    solution: Any,
    instance: Dict[str, Any],
) -> float:
    """
    Objective: minimize total travel distance (Euclidean).
    Assumes feasibility has already been checked (single flat route).
    """
    if not isinstance(solution, list):
        raise ValueError("Solution must be a flat route list.")

    if any(isinstance(x, list) for x in solution):
        raise ValueError("Nested routes are not supported for PDP objective.")

    coords = instance["coords"]
    dist = 0.0
    for i in range(1, len(solution)):
        dist += _euclid(coords[solution[i - 1]], coords[solution[i]])
    return float(dist)
