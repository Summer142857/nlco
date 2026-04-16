from typing import Any, Dict, List, Set, Tuple
import math
import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # permutation / tour


def mlp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse MLP instance from `instance_variant`.

    Expected `instance_variant` format (already labeled), e.g.:
      {
        "problem_type": "MLP",
        "num_nodes": 13,
        "nodes": [
          {"id": "A", "x": 78, "y": 0},
          {"id": "B", "x": 56, "y": 58},
          ...
        ],
        "depot": "A"
      }

    Returns:
      {
        "nodes": {id: (x, y)},
        "depot": depot_id
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for MLP.")

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
        else:
            raise ValueError("MLP instance has no nodes and no depot.")

    return {"nodes": nodes, "depot": depot}



def mlp_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check MLP feasibility + unified violation pattern.

    Patterns for MLP (per taxonomy Global_1):
      - FormatError: invalid format/types/unknown labels/instance mismatch
      - Global1: tour/permutation violations (start/end depot, missing/duplicate nodes, depot inside)
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]
    depot = instance["depot"]
    node_set: Set[Any] = set(nodes.keys())

    if not isinstance(solution, list) or len(solution) < 2:
        return False, PATTERN_FORMAT, "Solution must be a list of node identifiers with length >= 2."

    if depot not in node_set:
        return False, PATTERN_FORMAT, f"Depot {depot!r} is not a valid node id in the instance."

    # ---- Global1: must start/end at depot ----
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

    return True, PATTERN_OK, "Feasible MLP tour."


def mlp_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    MLP objective (to MINIMIZE): sum of arrival times (latencies).

    Let cumulative distance after traversing edge i be T_i.
    For a tour [depot, v1, v2, ..., vk, depot],
    arrival times are: T at v1, v2, ..., vk (NON-depot nodes).
    We DO NOT include the final return-to-depot arrival time.

    This matches your provided obj example (1522.1336...).
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]

    if not isinstance(solution, list) or len(solution) < 2:
        return float("inf")

    cumulative = 0.0
    total_latency = 0.0

    # traverse edges up to the last visited non-depot node
    # i.e., exclude the final edge (last_node -> depot) from latency sum.
    for u, v in zip(solution[:-2], solution[1:-1]):
        if u not in nodes or v not in nodes:
            return float("inf")
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        cumulative += math.hypot(x1 - x2, y1 - y2)
        total_latency += cumulative

    return float(total_latency)
