from typing import Any, Dict, List, Set, Tuple
import math
import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # permutation / tour
PATTERN_GLOBAL2 = "Global2"  # budgeted subset (quota/budget constraint)


def pctsp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse PCTSP instance from `instance_variant`.

    Expected instance_variant format:
      {
        "problem_type": "PCTSP",
        "num_nodes": 6,
        "nodes": [
          {"id": 0, "x": 100, "y": 88, "prize": 0, "penalty": 0},
          ...
        ],
        "depot": 0,
        "required_prize": 150.0
      }

    Returns:
      {
        "nodes": {id: (x, y)},
        "prizes": {id: prize},
        "penalties": {id: penalty},
        "depot": depot_id,
        "required_prize": float
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for PCTSP.")

    nodes_raw = inst_var.get("nodes", [])
    nodes: Dict[Any, Tuple[float, float]] = {}
    prizes: Dict[Any, float] = {}
    penalties: Dict[Any, float] = {}

    for n in nodes_raw:
        node_id = n["id"]
        nodes[node_id] = (float(n["x"]), float(n["y"]))
        prizes[node_id] = float(n.get("prize", 0.0))
        penalties[node_id] = float(n.get("penalty", 0.0))

    depot = inst_var.get("depot", None)
    if depot is None:
        if nodes:
            depot = next(iter(nodes.keys()))
        else:
            raise ValueError("PCTSP instance has no nodes and no depot.")

    required_prize = float(inst_var.get("required_prize", 0.0))

    return {
        "nodes": nodes,
        "prizes": prizes,
        "penalties": penalties,
        "depot": depot,
        "required_prize": required_prize,
    }



def pctsp_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check PCTSP feasibility + unified violation pattern.

    Patterns for PCTSP (per taxonomy Global_{1,2}):
      - FormatError: invalid type/length, unknown labels, instance mismatch
      - Global1: tour violations (start/end depot, duplicates, depot in interior)
      - Global2: prize quota not met (collected_prize < required_prize)
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]
    prizes: Dict[Any, float] = instance["prizes"]
    depot = instance["depot"]
    required_prize = float(instance["required_prize"])

    node_set: Set[Any] = set(nodes.keys())

    # ---- Format checks ----
    if not isinstance(solution, list) or len(solution) < 2:
        return False, PATTERN_FORMAT, "Solution must be a list with at least start and end depot."

    if depot not in node_set:
        return False, PATTERN_FORMAT, f"Depot {depot!r} is not a valid node id in the instance."

    # ---- Global1: start/end at depot ----
    if solution[0] != depot or solution[-1] != depot:
        return False, PATTERN_GLOBAL1, f"Tour must start and end at the depot {depot!r}."

    interior = solution[1:-1]

    # ---- FormatError: unknown labels ----
    unknown = set(interior) - node_set
    if unknown:
        return False, PATTERN_FORMAT, f"Solution contains unknown node labels: {unknown!r}"

    # ---- Global1: depot not allowed inside ----
    if depot in interior:
        return False, PATTERN_GLOBAL1, f"Depot {depot!r} must not appear in the interior of the tour."

    # ---- Global1: no duplicates ----
    if len(interior) != len(set(interior)):
        return False, PATTERN_GLOBAL1, "Some nodes are visited more than once in the interior of the tour."

    # ---- Global2: prize quota ----
    collected_prize = 0.0
    for v in interior:
        collected_prize += float(prizes.get(v, 0.0))

    if collected_prize + 1e-9 < required_prize:
        return (
            False,
            PATTERN_GLOBAL2,
            f"Collected prize {collected_prize:.3f} is less than required_prize {required_prize:.3f}.",
        )

    return True, PATTERN_OK, "Feasible PCTSP tour."


def pctsp_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    PCTSP objective (to MINIMIZE):
      objective = route_length + sum(penalty of unvisited non-depot nodes)

    route_length: Euclidean length along consecutive edges in the tour.
    unvisited penalty: sum of penalties for nodes not visited in the interior,
                       excluding depot.
    """
    nodes: Dict[Any, Tuple[float, float]] = instance["nodes"]
    penalties: Dict[Any, float] = instance["penalties"]
    depot = instance["depot"]

    if not isinstance(solution, list) or len(solution) < 2:
        return float("inf")

    # route length
    route_length = 0.0
    for u, v in zip(solution[:-1], solution[1:]):
        if u not in nodes or v not in nodes:
            return float("inf")
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        route_length += math.hypot(x1 - x2, y1 - y2)

    # unvisited penalties (exclude depot)
    visited_set = set(solution[1:-1])
    unvisited_penalty = 0.0
    for nid in nodes.keys():
        if nid == depot:
            continue
        if nid not in visited_set:
            unvisited_penalty += float(penalties.get(nid, 0.0))

    return float(route_length + unvisited_penalty)
