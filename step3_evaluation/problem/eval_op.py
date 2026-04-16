import math
from typing import Any, Dict, List, Set, FrozenSet, Tuple

import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # permutation / tour (route structure)
PATTERN_GLOBAL2 = "Global2"  # budgeted subset (length budget)


def op_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Orienteering Problem (OP) instance from `instance_variant`.

    Returns:
      {
        "nodes": {id: (x, y)},
        "prizes": {id: prize},
        "depot": depot_id,
        "max_length": float
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for OP.")

    nodes_raw = inst_var.get("nodes", [])

    nodes: Dict[Any, Tuple[float, float]] = {}
    prizes: Dict[Any, float] = {}

    for n in nodes_raw:
        node_id = n["id"]
        nodes[node_id] = (float(n["x"]), float(n["y"]))
        prizes[node_id] = float(n.get("prize", 0.0))

    depot = inst_var.get("depot", None)
    if depot is None:
        raise ValueError("OP instance missing depot.")

    max_length = float(inst_var.get("max_length", float("inf")))

    return {
        "nodes": nodes,
        "prizes": prizes,
        "depot": depot,
        "max_length": max_length,
    }
    
    

def op_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check OP feasibility + unified violation pattern.

    Patterns for OP (per taxonomy Global_{1,2}):
      - FormatError: invalid type/length, unknown labels, instance mismatch
      - Global1: tour/path structure violations (start/end depot, repeats, depot inside)
      - Global2: length budget violation (tour length > max_length)
    """
    nodes = instance["nodes"]          # Dict[node_id, (x,y)]
    depot = instance["depot"]
    max_length = float(instance["max_length"])
    node_set: Set[Any] = set(nodes.keys())

    # ---- Format checks ----
    if not isinstance(solution, list) or len(solution) < 2:
        return False, PATTERN_FORMAT, "Solution must be a list with at least start and end."

    if depot not in node_set:
        return False, PATTERN_FORMAT, f"Depot {depot!r} is not a valid node id in the instance."

    # ---- Global1: start/end at depot ----
    if solution[0] != depot or solution[-1] != depot:
        return False, PATTERN_GLOBAL1, f"Tour must start and end at depot {depot!r}."

    interior = solution[1:-1]

    # ---- FormatError: unknown nodes ----
    unknown = set(interior) - node_set
    if unknown:
        return False, PATTERN_FORMAT, f"Unknown node labels: {unknown!r}"

    # ---- Global1: depot not allowed inside ----
    if depot in interior:
        return False, PATTERN_GLOBAL1, "Depot must not appear in the interior of the tour."

    # ---- Global1: no duplicates in interior ----
    if len(interior) != len(set(interior)):
        return False, PATTERN_GLOBAL1, "Some nodes are visited more than once."

    # ---- Global2: length budget ----
    total_length = 0.0
    try:
        for u, v in zip(solution[:-1], solution[1:]):
            x1, y1 = nodes[u]
            x2, y2 = nodes[v]
            total_length += math.hypot(x1 - x2, y1 - y2)
    except Exception as e:
        # If coordinates missing/bad types, treat as format/instance mismatch
        return False, PATTERN_FORMAT, f"Could not compute tour length due to invalid node coordinates: {e!r}"

    if total_length > max_length + 1e-6:
        return False, PATTERN_GLOBAL2, f"Tour length {total_length:.3f} exceeds max_length {max_length:.3f}."

    return True, PATTERN_OK, "Feasible OP tour."



def op_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    OP objective: total collected prize.

    Sum of prizes for all visited non-depot nodes.
    """
    prizes = instance["prizes"]
    depot = instance["depot"]

    if not isinstance(solution, list) or len(solution) < 2:
        return float("-inf")

    total_prize = 0.0
    for node in solution[1:-1]:
        if node == depot:
            continue
        total_prize += prizes.get(node, 0.0)

    return float(total_prize)
