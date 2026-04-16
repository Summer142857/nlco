from typing import Any, Dict, List, Set, FrozenSet, Tuple
import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL5 = "Global5"  # partitioning
PATTERN_GLOBAL8 = "Global8"  # local graph labeling (adjacency constraints)



def gcp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse GCP instance from `instance_variant` in label space.

    Expected instance_variant format:
      {
        "num_nodes": 23,
        "num_edges": 100,
        "edges": [{"u": 2, "v": 18}, ...]
      }

    Returns:
      {
        "nodes": set of node labels,
        "edges": set of frozenset({u,v}) undirected
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for GCP.")

    edges_raw = inst_var.get("edges", [])
    nodes: Set[Any] = set()
    edges: Set[FrozenSet[Any]] = set()

    for e in edges_raw:
        u = e["u"]
        v = e["v"]
        if u == v:
            continue
        nodes.add(u)
        nodes.add(v)
        edges.add(frozenset({u, v}))

    return {"nodes": nodes, "edges": edges}


def gcp_check_feasibility(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check GCP feasibility + violation pattern.

    Patterns for GCP (per taxonomy Global_{5,8}):
      - FormatError: invalid format/types/unknown labels/instance inconsistency
      - Global5: partitioning violation (duplicate node / missing node)
      - Global8: adjacency labeling violation (edge endpoints same color)
    """
    nodes: Set[Any] = instance["nodes"]
    edges: Set[FrozenSet[Any]] = instance["edges"]

    # ---- Format checks ----
    if not isinstance(solution, list) or len(solution) == 0:
        return False, PATTERN_FORMAT, "Solution must be a non-empty list of color classes."

    assignment: Dict[Any, int] = {}

    for color_idx, color_class in enumerate(solution):
        if not isinstance(color_class, list):
            return False, PATTERN_FORMAT, f"Color class {color_idx} must be a list of node labels."
        for v in color_class:
            # ---- Global5: each node in exactly one class (no duplicates) ----
            if v in assignment:
                return False, PATTERN_GLOBAL5, f"Node {v!r} appears in more than one color class."
            assignment[v] = color_idx

    # ---- FormatError: unknown labels ----
    unknown = set(assignment.keys()) - nodes
    if unknown:
        return False, PATTERN_FORMAT, f"Solution contains unknown node labels: {unknown!r}"

    # ---- Global5: missing nodes (must color all nodes) ----
    missing = nodes - set(assignment.keys())
    if missing:
        return False, PATTERN_GLOBAL5, f"Some nodes are missing from the coloring: {missing!r}"

    # ---- Global8: edge constraint ----
    for e in edges:
        if len(e) != 2:
            continue
        u, v = tuple(e)
        cu = assignment.get(u, None)
        cv = assignment.get(v, None)
        # Should not happen if missing/unknown handled, but keep guard
        if cu is None or cv is None:
            return False, PATTERN_FORMAT, f"Internal error: unassigned endpoint in edge {set(e)!r}."
        if cu == cv:
            return False, PATTERN_GLOBAL8, f"Invalid coloring: edge {u!r}-{v!r} has same color {cu}."

    return True, PATTERN_OK, "Feasible GCP coloring."


def gcp_objective_model(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> float:
    """
    GCP objective (to MINIMIZE): number of colors used.
    We count non-empty color classes for robustness.
    """
    if not isinstance(solution, list):
        return float("inf")
    # Count non-empty classes (empty classes shouldn't matter, but often indicate a bad output)
    num_colors = sum(1 for cls in solution if isinstance(cls, list) and len(cls) > 0)
    return float(num_colors)
